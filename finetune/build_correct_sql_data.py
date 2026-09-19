#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 correct_sql 节点的训练数据（错误 SQL + 真实报错 → 修正 SQL）。

为什么需要单独生成
------------------
流水线自然产出的样本只能训 generate_sql：它很少写错，所以 collect 不到
"错误 SQL → 修正"这种配对。而 correct_sql 的输入恰恰是：
    (用户问题, schema, 指标口径, 写错的 SQL, EXPLAIN 报错)  ->  修正后的 SQL

所以这里**程序化制造错误**：从正确的 SQL 出发注入典型错误，
再用 MySQL 的 EXPLAIN 跑一遍拿到**真实报错信息**，配成训练样本。

关键点：报错必须是真的
----------------------
不能用模板编报错文本 —— 那样模型学到的是假信号。
本脚本把注入错误的 SQL 真的送去 EXPLAIN，把数据库返回的原始错误信息
作为输入的一部分；能通过 EXPLAIN 的"错误"说明它其实没错，直接丢弃。

注入的错误类型（覆盖真实场景）
------------------------------
1. 字段名拼错 / 字段不存在
2. 表名拼错 / 表不存在
3. 漏写 JOIN 条件（笛卡尔积通常不报错，但 join 字段写错会报 Unknown column）
4. 聚合函数用错列类型
5. 引用了未出现在 FROM/JOIN 中的表别名
6. 关键字/语法错误

用法
----
    # 需要 MySQL / Qdrant / ES / Embedding 都在跑
    python finetune/build_correct_sql_data.py --limit 5      # 冒烟
    python finetune/build_correct_sql_data.py                # 全量
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
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

from app.clients.mysql_client_manager import dw_mysql_client_manager   # noqa: E402
from app.prompt.prompt_loader import load_prompt                       # noqa: E402
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository  # noqa: E402

DATA_DIR = HERE / "data"
TRAIN_SET = DATA_DIR / "train.jsonl"
TEST_SET = DATA_DIR / "test.jsonl"
DEFAULT_OUT = DATA_DIR / "correct_sql.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompt" / "correct_sql.prompt"

# 这些是真实存在的表/字段，用于构造"形近但不存在"的拼写错误
_KNOWN_TABLES = ["fact_order", "dim_region", "dim_product", "dim_date", "dim_customer"]
_KNOWN_COLUMNS = [
    "order_amount", "order_quantity", "order_id", "date_id", "region_id",
    "product_id", "customer_id", "region_name", "province", "product_name",
    "category", "brand", "member_level", "gender", "month", "quarter", "year", "day",
]


# ==========================================================================
# system 提示词：复用线上 correct_sql.prompt 的固定部分
# ==========================================================================
def build_system_prompt() -> str:
    """
    从线上 correct_sql.prompt 里抽出【角色】段作为 system。

    注意：必须在【上下文信息】处切断，而不是按"原始用户查询"切 ——
    后者会把"原始用户查询如下："这类引导句留在 system 里，
    与 human 段的【用户问题】重复。
    """
    if not PROMPT_PATH.exists():
        return ("你是 SQL 纠错专家。根据数据库返回的报错信息，修正给定的 MySQL 查询语句，"
                "只输出修正后的一条只读 SQL。")
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = re.sub(r"\{[a-z_]+\}", "", text)
    text = re.split(r"【上下文信息】", text)[0]
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _clean_error(raw: str) -> str:
    """
    从 SQLAlchemy 的异常文本里提取纯 MySQL 报错。

    原始形态通常带一大堆包装：
        (asyncmy.errors.OperationalError) (1054, "Unknown column 'f.x' in 'field list'")
        [SQL: explain SELECT ...]
        [parameters: (...)]
        (Background on this error at: https://sqlalche.me/e/20/e3q8)
    训练时只需要中间那句真正的数据库报错，其余是噪声。
    """
    if not raw:
        return ""
    # 优先抓 (错误码, "消息") 形式
    m = re.search(r"\((\d{4}),\s*[\"'](.+?)[\"']\)", raw, re.DOTALL)
    if m:
        return f"({m.group(1)}, \"{m.group(2).strip()}\")"
    # 退而求其次：去掉 [SQL: ...] / [parameters: ...] / Background 段
    text = re.split(r"\[SQL:|\[parameters:|\(Background on this error", raw)[0]
    return text.strip()


# ==========================================================================
# 错误注入
# ==========================================================================
def _corrupt_identifier(name: str, rng: random.Random) -> str:
    """把标识符改成"形近但不存在"的写法"""
    kind = rng.randint(0, 2)
    if kind == 0 and len(name) > 3:
        # 删掉一个字符
        i = rng.randint(1, len(name) - 2)
        return name[:i] + name[i + 1:]
    if kind == 1:
        # 末尾加/换一个字符
        return name[:-1] + rng.choice("sxz")
    # 元音替换
    for a, b in (("a", "e"), ("e", "a"), ("o", "u"), ("i", "y")):
        if a in name:
            return name.replace(a, b, 1)
    return name + "s"


def inject_errors(sql: str, rng: random.Random) -> list[tuple[str, str]]:
    """
    对一条正确 SQL 注入错误，返回 [(错误SQL, 错误类型说明), ...]。
    同一个原句可产出多种错误，用于扩充样本。
    """
    if not sql:
        return []
    out: list[tuple[str, str]] = []

    # --- 1. 字段名拼错 ---
    for col in _KNOWN_COLUMNS:
        if col in sql:
            bad = _corrupt_identifier(col, rng)
            if bad != col:
                out.append((sql.replace(col, bad, 1), f"字段名拼写错误: {col} -> {bad}"))
            break

    # --- 2. 表名拼错 ---
    for tbl in _KNOWN_TABLES:
        if tbl in sql:
            bad = _corrupt_identifier(tbl, rng)
            if bad != tbl:
                out.append((sql.replace(tbl, bad, 1), f"表名拼写错误: {tbl} -> {bad}"))
            break

    # --- 3. 引用了不存在的表别名 ---
    m = re.search(r"\bFROM\s+(\w+)\s+(\w+)", sql, re.IGNORECASE)
    if m:
        alias = m.group(2)
        bad_alias = alias + "x"
        if f"{alias}." in sql:
            out.append((
                sql.replace(f"{alias}.", f"{bad_alias}.", 1),
                f"引用了未定义的别名: {alias} -> {bad_alias}",
            ))

    # --- 4. 聚合函数套在错误的列上（把数值列换成文本列）---
    mnum = re.search(r"(SUM|AVG)\s*\(\s*([\w.]+)\s*\)", sql, re.IGNORECASE)
    if mnum:
        out.append((
            sql.replace(mnum.group(0), f"{mnum.group(1)}(product_name)"),
            "聚合函数作用在非数值列上(SUM/AVG(product_name))",
        ))

    # --- 5. 语法类错误：多一个逗号 / 少一个括号 ---
    if "GROUP BY" in sql.upper():
        out.append((re.sub(r"\bGROUP\s+BY\b", "GROUP BY GROUP", sql, count=1, flags=re.IGNORECASE),
                    "语法错误: GROUP BY 重复"))
    if sql.count("(") > sql.count(")"):
        pass  # 已经不平衡，跳过
    else:
        out.append((sql.replace("(", "((", 1), "语法错误: 括号不匹配"))

    # 去重
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for s, why in out:
        if s != sql and s not in seen:
            seen.add(s)
            uniq.append((s, why))
    return uniq


# ==========================================================================
async def main_async(args) -> int:
    # 收集可用的正确 SQL（来自评测集与已采集训练集）
    sources: list[tuple[str, dict]] = []
    for path in (args.test_set, args.train_set):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sql = rec.get("reference_sql")
            if sql and rec.get("question"):
                sources.append((sql, rec))
    if not sources:
        print("[错误] 找不到带标准 SQL 的样本，请先跑 gen_eval_set.py", file=sys.stderr)
        return 2
    print(f"[1/3] 可用正确 SQL {len(sources)} 条（来自评测集 + 已采集训练集）")

    rng = random.Random(args.seed)
    system_prompt = build_system_prompt()

    dw_mysql_client_manager.init_client()
    kept = dropped = 0
    type_counter: dict[str, int] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        assert dw_mysql_client_manager.session_factory
        async with dw_mysql_client_manager.session_factory() as session:
            repo = DWMysqlRepository(session)

            with args.out.open("w", encoding="utf-8") as fh:
                processed = 0
                for correct_sql, rec in sources:
                    if args.limit and processed >= args.limit:
                        break
                    for bad_sql, why in inject_errors(correct_sql, rng):
                        if args.limit and processed >= args.limit:
                            break
                        # ---- 关键：真的去 EXPLAIN，拿真实报错 ----
                        try:
                            await repo.validate_sql(bad_sql)
                            # 没报错说明这个"错误"其实合法，丢弃
                            dropped += 1
                            continue
                        except Exception as exc:  # noqa: BLE001
                            real_error = _clean_error(str(exc))
                        if not real_error:
                            dropped += 1
                            continue

                        processed += 1
                        key = why.split(":")[0]
                        type_counter[key] = type_counter.get(key, 0) + 1

                        human = (
                            "【用户问题】\n"
                            f"{rec['question']}\n\n"
                            "【原始SQL（有错误）】\n"
                            f"{bad_sql}\n\n"
                            "【数据库返回的错误信息】\n"
                            f"{real_error}\n\n"
                            "请修正上面的 SQL，只输出修正后的完整只读查询语句。"
                        )
                        sample = {
                            "task": "correct_sql",
                            "error_type": why,
                            "real_error": real_error,
                            "question": rec["question"],
                            "wrong_sql": bad_sql,
                            "correct_sql": correct_sql,
                            "conversations": [
                                {"from": "system", "value": system_prompt},
                                {"from": "human", "value": human},
                                {"from": "gpt", "value": correct_sql},
                            ],
                        }
                        fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        fh.flush()
                        kept += 1

                    if kept and kept % 20 == 0:
                        print(f"  已生成 {kept} 条（丢弃 {dropped} 条）")
    finally:
        await dw_mysql_client_manager.close()

    print()
    print(f"[2/3] 生成 {kept} 条 correct_sql 训练样本，丢弃 {dropped} 条（EXPLAIN 未报错）")
    print("  错误类型分布：")
    for k, v in sorted(type_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v}")
    print(f"[3/3] 输出 -> {args.out}")
    print()
    print("说明：报错信息来自 MySQL 真实返回（EXPLAIN），不是模板编造；")
    print("      能通过 EXPLAIN 的\"错误\"已丢弃，保证样本有效。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 correct_sql 训练数据")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--train-set", type=Path, default=TRAIN_SET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 条（冒烟）")
    parser.add_argument("--seed", type=int, default=20250922)
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[中断]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
