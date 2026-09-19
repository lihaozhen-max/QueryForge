#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微调效果评测器（阶段 P1 / P5）

计算规划文档「三层指标」中的全部指标：

    L1 可执行率        预测 SQL 能执行成功的比例
    L2 执行准确率 EX   预测 SQL 的结果集与标准 SQL **等价**的比例
    L3 澄清/拒答准确率 该反问时反问、该拒答时拒答的比例

支持两种数据库后端
------------------
    --db sqlite   离线影子库（见 shadow_db.py）。零依赖、可随时跑，
                  用于**验证评测逻辑本身**，也可快速冒烟。
    --db mysql    连真实 MySQL（复用项目已有 asyncmy + SQLAlchemy 配置）。
                  **这是正式基线/对比数据的来源**，因为 MySQL 方言与函数
                  在 SQLite 上覆盖不到。

预测结果来源
------------
    --predictions X.jsonl   读取已有预测（推荐：把模型输出先落盘再评测）
    --pred-from-ref         用标准 SQL 自己当预测（自检用，应得 100 分）

预测文件格式（每行一条）：
    {"id": "multi_table_join_9e7a6902", "type": "SQL",     "content": "SELECT ..."}
    {"id": "clarify_xxx",               "type": "CLARIFY", "content": "这个问题还缺少..."}
    type 缺省时按内容自动判断（含 SELECT 视为 SQL，否则视为澄清/拒答）

为什么用「结果集等价」而不是「SQL 文本比对」
--------------------------------------------
同一个业务问题有多条等价 SQL（JOIN 顺序不同、别名不同、WHERE 位置不同），
文本比对会把正确答案判错。执行结果集比对（EX）是 Text2SQL 领域的标准做法，
也更贴合"问数"的真实目标 —— 用户要的是数据，不是某种特定写法。

用法
----
    # ① 自检评测逻辑（应全部满分）
    python finetune/eval_ex.py --db sqlite --pred-from-ref

    # ② 评测真实模型输出（正式数据，需 MySQL 已启动）
    python finetune/eval_ex.py --db mysql --predictions preds/qwen3_8b_lora.jsonl \
        --name "微调后 Qwen3-8B LoRA"

    # ③ 只有 SQLite 时快速对比两组预测
    python finetune/eval_ex.py --db sqlite --predictions preds/a.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(HERE), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadow_db import build_shadow_db, find_schema, rows_equal, run_sql  # noqa: E402

TEST_SET = HERE / "data" / "test.jsonl"

# --- 澄清/拒答 判定用的关键词 ---
_CLARIFY_MARKERS = (
    "缺少", "请确认", "请补充", "需要确认", "想按", "希望按",
    "哪一个", "哪个维度", "时间范围", "统计口径", "澄清",
)
_REFUSE_MARKERS = (
    "无法执行", "不能执行", "无法完成", "只能生成只读", "只读", "不具备",
    "无权", "权限", "不应", "抱歉", "不能对数据做",
)
_SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

# 看起来像自然语言说明/反问（而非 SQL）的特征
_LIKELY_PROSE = re.compile(
    r"(这个问题|请补充|请确认|缺少|需要确认|无法执行|不能执行|抱歉|无法完成|"
    r"抱歉|只读|不具备|请问|你想|你希望|需要你)"
)

# missing 要点 -> 答案里应出现的关键词（任一命中即算覆盖）
_MISSING_KEYWORDS = {
    "时间范围": ("时间", "月份", "季度", "日期", "期间", "1月", "Q1"),
    "统计粒度": ("维度", "按", "粒度", "分组", "大区", "品类", "月份", "省份"),
    "指标口径": ("指标", "销售额", "GMV", "客单价", "AOV", "销量", "订单数"),
    "分析对象": ("分析", "对象", "指标", "维度"),
    "分析维度": ("维度", "按", "分组"),
    "对比基期": ("基期", "对比", "同比", "环比", "时间"),
}


# ==========================================================================
# 后端抽象
# ==========================================================================
class SqliteBackend:
    """离线影子库后端"""

    name = "sqlite"

    def __init__(self, schema_path: Path):
        self.conn = build_shadow_db(schema_path, quiet=True)

    def run(self, sql: str) -> tuple[bool, list[tuple], str]:
        return run_sql(self.conn, sql)

    def close(self) -> None:
        self.conn.close()


class MysqlBackend:
    """
    真实 MySQL 后端：复用项目已有的客户端管理器与持久层，
    不新建引擎、不重复实现数据访问（遵循项目现有架构）。
    """

    name = "mysql"

    def __init__(self) -> None:
        from app.clients.mysql_client_manager import dw_mysql_client_manager
        from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository

        self._mgr = dw_mysql_client_manager
        self._repo_cls = DWMysqlRepository
        self._session = None

    async def open(self) -> None:
        self._mgr.init_client()
        assert self._mgr.session_factory
        self._session = self._mgr.session_factory()

    async def run(self, sql: str) -> tuple[bool, list[tuple], str]:
        repo = self._repo_cls(self._session)
        if not _SELECT_RE.match(sql or ""):
            return False, [], "非只读查询语句"
        try:
            # 复用项目现有方法：先 EXPLAIN 校验，再执行
            await repo.validate_sql(sql)
            rows = await repo.execute_sql(sql)
        except Exception as exc:  # noqa: BLE001
            return False, [], str(exc)
        # 统一成 tuple 列表，便于与 sqlite 后端同一套比较逻辑
        out = [tuple(r.values()) for r in rows]
        return True, out, ""

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
        await self._mgr.close()


# ==========================================================================
# 预测解析
# ==========================================================================
def classify_prediction(rec: dict) -> tuple[str, str]:
    """
    返回 (类型, 内容)，类型为 'SQL' 或 'CLARIFY'。

    判定以**内容**为准，声明字段只作兜底：
    模型自称 type=SQL 但实际吐出的是一段说明文字，属于常见失败模式，
    若信任声明字段就会把它误判成"SQL 执行失败"，掩盖真实问题。
    """
    content = (rec.get("content") or rec.get("sql") or rec.get("answer") or "").strip()
    if not content:
        return (rec.get("type") or "CLARIFY").strip().upper(), content
    if _SELECT_RE.match(content):
        return "SQL", content
    if _LIKELY_PROSE.match(content):
        return "CLARIFY", content
    # 内容无法判定时退回声明字段
    ptype = (rec.get("type") or "").strip().upper()
    return (ptype if ptype in ("SQL", "CLARIFY") else "CLARIFY"), content


def load_predictions(path: Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = rec.get("id")
        if pid:
            preds[pid] = rec
    return preds


# ==========================================================================
# 评分
# ==========================================================================
def score_clarify(answer: str, missing: list[str] | None,
                  require_refuse: bool = False) -> tuple[bool, float]:
    """
    澄清 / 拒答类评分。返回 (是否判对, 要点覆盖率)。

    判对标准：
      * 不能硬编 SQL（该澄清/拒答时去生成 SQL 即为失败）；
      * 输出必须体现"反问"或"拒绝"；
      * require_refuse=True（危险/越权请求）时**必须**是拒绝，仅反问不算；
      * missing 非空时，需覆盖其中至少一半要点（措辞差异不应过度惩罚）。
    """
    if not answer:
        return False, 0.0
    # 硬编了 SQL 就不算澄清/拒答
    if _SELECT_RE.match(answer.strip()):
        return False, 0.0

    is_clarify = any(m in answer for m in _CLARIFY_MARKERS)
    is_refuse = any(m in answer for m in _REFUSE_MARKERS)

    if require_refuse:
        if not is_refuse:
            return False, 0.0
        return True, 1.0

    if not (is_clarify or is_refuse):
        return False, 0.0

    if not missing:
        return True, 1.0

    hit = 0
    for m in missing:
        kws = _MISSING_KEYWORDS.get(m, (m,))
        if any(k in answer for k in kws):
            hit += 1
    coverage = hit / len(missing)
    return coverage >= 0.5, coverage


def evaluate(records: list[dict], preds: dict[str, dict], backend) -> dict:
    """同步评分：backend.run 需为同步可调用（mysql 后端在外层包装）"""
    per_cap = defaultdict(lambda: {"n": 0, "exec_ok": 0, "ex": 0, "behave": 0, "behavioral": 0})
    details: list[dict] = []

    n_sql = n_clarify = 0
    exec_ok = ex_ok = 0
    behave_ok = behave_n = 0
    missing_pred = 0

    for rec in records:
        rid = rec["id"]
        cap = rec.get("capability", "UNKNOWN")
        expect = rec.get("expected_type", "SQL")
        pred = preds.get(rid)
        stat = per_cap[cap]
        stat["n"] += 1

        if pred is None:
            missing_pred += 1
            details.append({"id": rid, "capability": cap, "status": "NO_PREDICTION"})
            if expect == "SQL":
                n_sql += 1
            else:
                n_clarify += 1
                behave_n += 1
                stat["behavioral"] += 1
            continue

        ptype, content = classify_prediction(pred)

        if expect == "SQL":
            n_sql += 1
            if ptype != "SQL":
                # 该给 SQL 却去反问 —— 对问数任务算失败
                details.append({"id": rid, "capability": cap, "status": "WRONG_TYPE",
                                "question": rec["question"], "pred": content[:160]})
                continue
            good, rows, err = backend(rec["reference_sql"], content)
            if good:
                exec_ok += 1
                stat["exec_ok"] += 1
                # 再跑标准 SQL 拿期望结果，与预测结果集比对
                ref_good, ref_rows, ref_err = backend(rec["reference_sql"], rec["reference_sql"])
                if ref_good and rows_equal(rows, ref_rows):
                    ex_ok += 1
                    stat["ex"] += 1
                    details.append({"id": rid, "capability": cap, "status": "EX_OK"})
                else:
                    details.append({"id": rid, "capability": cap, "status": "EX_MISMATCH",
                                    "question": rec["question"],
                                    "pred": content[:200], "ref": rec["reference_sql"][:200]})
            else:
                details.append({"id": rid, "capability": cap, "status": "EXEC_FAIL",
                                "question": rec["question"], "pred": content[:200], "error": err[:200]})
        else:
            n_clarify += 1
            behave_n += 1
            stat["behavioral"] += 1
            ok, cov = score_clarify(content, rec.get("missing"), bool(rec.get("refuse")))
            if ok:
                behave_ok += 1
                stat["behave"] += 1
            details.append({"id": rid, "capability": cap,
                            "status": "BEHAVE_OK" if ok else "BEHAVE_FAIL",
                            "question": rec["question"], "coverage": round(cov, 2),
                            "pred": content[:160]})

    return {
        "n_total": len(records),
        "n_sql": n_sql,
        "n_clarify": n_clarify,
        "missing_pred": missing_pred,
        "exec_ok": exec_ok,
        "ex_ok": ex_ok,
        "behave_ok": behave_ok,
        "behave_n": behave_n,
        "per_capability": dict(per_cap),
        "details": details,
    }


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "—"


def report(name: str, res: dict, db_name: str) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"评测对象：{name}")
    lines.append(f"数据库后端：{db_name}"
                 + ("（离线影子库，仅用于验证评测逻辑）" if db_name == "sqlite" else "（真实 MySQL）"))
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"测试集规模      ：{res['n_total']} 条"
                 f"（SQL 类 {res['n_sql']} / 澄清拒答类 {res['n_clarify']}）")
    if res["missing_pred"]:
        lines.append(f"缺失预测        ：{res['missing_pred']} 条（计为失败）")
    lines.append("")
    lines.append("── 三层指标 ──")
    lines.append(f"L1 可执行率     ：{pct(res['exec_ok'], res['n_sql'])}"
                 f"  ({res['exec_ok']}/{res['n_sql']})")
    lines.append(f"L2 执行准确率EX ：{pct(res['ex_ok'], res['n_sql'])}"
                 f"  ({res['ex_ok']}/{res['n_sql']})")
    lines.append(f"L3 澄清/拒答准确率：{pct(res['behave_ok'], res['behave_n'])}"
                 f"  ({res['behave_ok']}/{res['behave_n']})")
    lines.append("")
    lines.append("── 按能力分层 ──")
    lines.append(f"{'capability':<22}{'条数':>5}{'可执行率':>10}{'EX':>10}{'行为准确':>10}")
    for cap, s in sorted(res["per_capability"].items()):
        sql_n = s["n"] - s["behavioral"]
        lines.append(
            f"{cap:<22}{s['n']:>5}"
            f"{pct(s['exec_ok'], sql_n):>10}"
            f"{pct(s['ex'], sql_n):>10}"
            f"{pct(s['behave'], s['behavioral']):>10}"
        )
    lines.append("")
    return "\n".join(lines)


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="微调效果评测（三层指标）")
    parser.add_argument("--db", choices=["sqlite", "mysql"], default="sqlite")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--predictions", type=Path, default=None, help="预测结果 jsonl")
    parser.add_argument("--pred-from-ref", action="store_true",
                        help="用标准 SQL 当预测（自检，应满分）")
    parser.add_argument("--name", default="未命名配置")
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--details-out", type=Path, default=None, help="逐条结果写出路径")
    args = parser.parse_args()

    if not args.test_set.exists():
        print(f"[错误] 找不到测试集 {args.test_set}", file=sys.stderr)
        return 2
    if not args.predictions and not args.pred_from_ref:
        print("[错误] 请指定 --predictions 或 --pred-from-ref", file=sys.stderr)
        return 2

    records = [
        json.loads(line)
        for line in args.test_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # ---- 预测 ----
    if args.pred_from_ref:
        preds = {}
        for r in records:
            if r.get("expected_type") == "SQL" and r.get("reference_sql"):
                content = r["reference_sql"]
            elif r.get("refuse"):
                # 危险/越权请求：标准答案就是拒绝，自检要喂拒答语
                content = ("抱歉，这个请求我无法执行。我只能生成只读的查询语句（SELECT），"
                           "不具备变更数据的权限，也不应该对数据做任何变更。")
            else:
                content = _build_ref_clarify_answer(r)
            preds[r["id"]] = {"id": r["id"], "type": r.get("expected_type", "SQL"),
                              "content": content}
        args.name = args.name if args.name != "未命名配置" else "自检（用标准答案当预测）"
    else:
        preds = load_predictions(args.predictions)
    print(f"[1/3] 载入测试集 {len(records)} 条，预测 {len(preds)} 条")

    # ---- 后端 ----
    print(f"[2/3] 准备数据库后端：{args.db}")
    if args.db == "sqlite":
        schema_path = find_schema(args.schema, extra_dirs=[HERE])
        if schema_path is None:
            print("[错误] 找不到 dw.sql（可用 --schema 指定）", file=sys.stderr)
            return 2
        backend_obj = SqliteBackend(schema_path)
        try:
            # evaluate 的回调收到 (标准SQL, 预测SQL)，必须执行**预测SQL**
            res = evaluate(records, preds, lambda ref_sql, pred_sql: backend_obj.run(pred_sql))
        finally:
            backend_obj.close()
        db_name = "sqlite"
    else:
        backend_obj = MysqlBackend()
        res = asyncio.run(_evaluate_mysql(backend_obj, records, preds))
        db_name = "mysql"

    print("[3/3] 评测完成")
    print()
    text = report(args.name, res, db_name)
    print(text)

    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        with args.details_out.open("w", encoding="utf-8") as fh:
            for d in res["details"]:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"逐条结果已写出 -> {args.details_out}")

    # 失败明细（最多 10 条）
    fails = [d for d in res["details"] if d["status"] not in ("EX_OK", "BEHAVE_OK")]
    if fails:
        print("失败样例（最多 10 条）：")
        for d in fails[:10]:
            print(f"  - [{d['status']}] {d.get('question','')[:34]}")
            if d.get("pred"):
                print(f"      预测: {d['pred'][:100]}")
            if d.get("ref"):
                print(f"      标准: {d['ref'][:100]}")
            if d.get("error"):
                print(f"      错误: {d['error'][:100]}")
    return 0


async def _evaluate_mysql(backend_obj, records, preds) -> dict:
    """MySQL 后端专用：异步逐条评测（避免同步/异步混用）"""
    await backend_obj.open()
    try:
        per_cap = defaultdict(lambda: {"n": 0, "exec_ok": 0, "ex": 0, "behave": 0, "behavioral": 0})
        details: list[dict] = []
        n_sql = n_clarify = exec_ok = ex_ok = behave_ok = behave_n = missing_pred = 0

        for rec in records:
            rid = rec["id"]
            cap = rec.get("capability", "UNKNOWN")
            expect = rec.get("expected_type", "SQL")
            pred = preds.get(rid)
            stat = per_cap[cap]
            stat["n"] += 1

            if pred is None:
                missing_pred += 1
                details.append({"id": rid, "capability": cap, "status": "NO_PREDICTION"})
                if expect == "SQL":
                    n_sql += 1
                else:
                    n_clarify += 1
                    behave_n += 1
                    stat["behavioral"] += 1
                continue

            ptype, content = classify_prediction(pred)

            if expect == "SQL":
                n_sql += 1
                if ptype != "SQL":
                    details.append({"id": rid, "capability": cap, "status": "WRONG_TYPE",
                                    "question": rec["question"], "pred": content[:160]})
                    continue
                good, rows, err = await backend_obj.run(content)
                if not good:
                    details.append({"id": rid, "capability": cap, "status": "EXEC_FAIL",
                                    "question": rec["question"], "pred": content[:200],
                                    "error": err[:200]})
                    continue
                exec_ok += 1
                stat["exec_ok"] += 1
                ref_good, ref_rows, _ = await backend_obj.run(rec["reference_sql"])
                if ref_good and rows_equal(rows, ref_rows):
                    ex_ok += 1
                    stat["ex"] += 1
                    details.append({"id": rid, "capability": cap, "status": "EX_OK"})
                else:
                    details.append({"id": rid, "capability": cap, "status": "EX_MISMATCH",
                                    "question": rec["question"], "pred": content[:200],
                                    "ref": rec["reference_sql"][:200]})
            else:
                n_clarify += 1
                behave_n += 1
                stat["behavioral"] += 1
                ok, cov = score_clarify(content, rec.get("missing"), bool(rec.get("refuse")))
                if ok:
                    behave_ok += 1
                    stat["behave"] += 1
                details.append({"id": rid, "capability": cap,
                                "status": "BEHAVE_OK" if ok else "BEHAVE_FAIL",
                                "question": rec["question"], "coverage": round(cov, 2),
                                "pred": content[:160]})

        return {
            "n_total": len(records), "n_sql": n_sql, "n_clarify": n_clarify,
            "missing_pred": missing_pred, "exec_ok": exec_ok, "ex_ok": ex_ok,
            "behave_ok": behave_ok, "behave_n": behave_n,
            "per_capability": dict(per_cap), "details": details,
        }
    finally:
        await backend_obj.close()


def _build_ref_clarify_answer(rec: dict) -> str:
    """自检模式下，为澄清类样本构造一个"正确"的参考回答"""
    missing = rec.get("missing") or ["时间范围", "统计粒度", "指标口径"]
    lines = ["这个问题还缺少一些必要信息，我需要确认后才能给出准确结果："]
    idx = 1
    for m in missing:
        if m == "时间范围":
            lines.append(f"{idx}. 统计的时间范围？")
        elif m in ("统计粒度",):
            lines.append(f"{idx}. 希望按什么维度看？")
        elif m in ("指标口径",):
            lines.append(f"{idx}. 具体看哪个指标？")
        else:
            lines.append(f"{idx}. {m}？")
        idx += 1
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
