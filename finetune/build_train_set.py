#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据采集脚本（阶段 P2）

做什么
------
拿 make_corpus.py 生成的问题，逐条喂给**现有的 LangGraph 流水线**，
把流水线检索出的 schema / 指标口径，以及它生成的 SQL 采集下来，
过滤掉执行不通过的样本，产出 LLaMA-Factory 可直接用的 sharegpt 训练集。

设计要点
--------
1. **不修改现有节点逻辑**：只从图外部用 stream_mode="values" 订阅，
   这与项目现有架构（服务层复用 compiled_graph）一致。
2. **答案必须执行校验**：标准 SQL 一律用 DWMysqlRepository.validate_sql()（EXPLAIN）
   与 execute_sql() 双重确认，跑不通的样本直接丢弃，绝不污染训练集。
3. **可断点续跑**：每采一条就落盘（jsonl 追加），中断后加 --resume 继续。
4. **零新增依赖**：只用项目已有依赖，不引入新库。

关于 system 提示词
    直接复用 prompt/generate_sql.prompt 的角色与约束部分，
    保证「训练时的输入格式」与「线上推理的输入格式」一致。
    提示词里与 schema/查询相关的占位符会被抽掉，避免与 human 段重复。

用法
----
    # 先确保 MySQL / Qdrant / ES / Embedding 服务都已启动
    python -m finetune.build_train_set --resume            # 从 finetune/data/corpus.txt 采集
    python -m finetune.build_train_set --limit 50          # 只跑 50 条做冒烟

    # 或直接指定问题文件
    python finetune/build_train_set.py --questions my.txt --out finetune/data/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.context import DataAgentContext            # noqa: E402
from app.agent.graph import compiled_graph                # noqa: E402
from app.agent.state import DataAgentState                # noqa: E402
from app.clients.embedding_client_manager import embedding_client_manager  # noqa: E402
from app.clients.es_client_manager import es_client_manager                # noqa: E402
from app.clients.mysql_client_manager import (                             # noqa: E402
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager        # noqa: E402
from app.core.log import logger                                           # noqa: E402
from app.repositories.es.value_es_repository import ValueESRepository      # noqa: E402
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository   # noqa: E402
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository  # noqa: E402
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository  # noqa: E402
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_QUESTIONS = DATA_DIR / "corpus.txt"
DEFAULT_OUT = DATA_DIR / "train.jsonl"
DEFAULT_REJECT = DATA_DIR / "train_rejected.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompt" / "generate_sql.prompt"

# 判定"该反问"的兜底特征：流水线目前没有澄清能力，遇到信息不全的问题也可能硬编 SQL。
# 采集阶段只做记录，不据此改标签；澄清样本在后续手工构造。
_CLARIFY_HINTS = re.compile(
    r"^(看看|帮我|给我|查一下|情况|数据|业务|最近|按).{0,6}(情况|怎么样|如何|怎样|看一下|统计一下|分析一下|报表)?$"
)


# ==========================================================================
# 提示词：从 generate_sql.prompt 抽出固定的角色与约束部分作为 system
# ==========================================================================
def build_system_prompt() -> str:
    """
    从线上提示词里抽取【角色】与【任务要求】两段做 system。

    这样做的目的：训练期与推理期的系统约束完全一致。
    若直接手写一份，后续线上改了 prompt 就会与训练分布漂移。
    """
    if not PROMPT_PATH.exists():
        return "你是掌柜问数的 SQL 生成助手。只依据给定的表结构与指标口径生成一条 MySQL 只读查询。"

    text = PROMPT_PATH.read_text(encoding="utf-8")
    # 去掉需要逐样本填充的占位符段，只保留角色与要求
    text = re.sub(r"\{[a-z_]+\}", "", text)
    # 去掉"用户查询"及其后的内容（那部分由 human 段承担）
    text = re.split(r"用户查询", text)[0]
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()).strip()


def build_human_prompt(state: dict, question: str) -> str:
    """
    构造 human 段：与线上 generate_sql 节点喂给模型的内容保持一致
    （table_infos / metric_infos / date_info / db_info + 用户问题）。
    """
    import yaml

    def dump(obj) -> str:
        if not obj:
            return "（无）"
        return yaml.dump(obj, allow_unicode=True, sort_keys=False).strip()

    return (
        "【可用数据表信息】\n"
        f"{dump(state.get('table_infos'))}\n\n"
        "【可参考的指标信息】\n"
        f"{dump(state.get('metric_infos'))}\n\n"
        "【当前时间信息】\n"
        f"{dump(state.get('date_info'))}\n\n"
        "【数据库环境】\n"
        f"{dump(state.get('db_info'))}\n\n"
        "【用户查询】\n"
        f"{question}"
    )


# ==========================================================================
# 单条问题：跑图 + 校验
# ==========================================================================
async def run_one(question: str, dw_session, meta_session) -> dict:
    """跑一次流水线，返回采集结果（不含判断，判断交给调用方）"""
    context = DataAgentContext(
        dw_mysql_repo=DWMysqlRepository(dw_session),
        meta_mysql_repo=MetaMysqlRepository(meta_session),
        value_es_repo=ValueESRepository(es_client_manager.client),
        column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
        metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
        embedding_client=embedding_client_manager.client,
    )
    state = DataAgentState(query=question, correct_sql_count=0)

    final: dict = {}
    # stream_mode="values" 会持续给出「当前累计状态」，最后一次即终态。
    # execute_sql 只往 stream_writer 写 result，不写 state，所以不能依赖 updates。
    async for snapshot in compiled_graph.astream(
        input=state,
        context=context,
        stream_mode="values",
    ):
        if isinstance(snapshot, dict):
            final = snapshot

    sql = (final.get("sql") or "").strip()
    error = final.get("error")
    corrected = final.get("correct_sql_count", 0)
    is_clarify_like = bool(_CLARIFY_HINTS.match(question.strip()))

    return {
        "question": question,
        "sql": sql,
        "error": error,
        "correct_sql_count": corrected,
        "is_clarify_like": is_clarify_like,
        "table_infos": final.get("table_infos") or [],
        "metric_infos": final.get("metric_infos") or [],
        "date_info": final.get("date_info") or {},
        "db_info": final.get("db_info") or {},
    }


async def check_sql_executable(sql: str, dw_repo: DWMysqlRepository) -> tuple[bool, str]:
    """用 EXPLAIN + 实际执行双重确认（复用项目现有方法，不新增逻辑）"""
    try:
        await dw_repo.validate_sql(sql)
    except Exception as exc:  # noqa: BLE001
        return False, f"EXPLAIN 失败: {exc}"
    try:
        rows = await dw_repo.execute_sql(sql)
    except Exception as exc:  # noqa: BLE001
        return False, f"执行失败: {exc}"
    if not rows:
        return False, "执行成功但结果为空"
    return True, ""


# ==========================================================================
# 主流程
# ==========================================================================
async def main_async(args) -> int:
    questions_path: Path = args.questions
    if not questions_path.exists():
        print(f"[错误] 找不到问题文件 {questions_path}", file=sys.stderr)
        print("       请先运行：python finetune/make_corpus.py", file=sys.stderr)
        return 2

    questions = [
        ln.strip()
        for ln in questions_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    # 断点续跑：跳过已采到的问题
    done: set[str] = set()
    if args.resume and args.out.exists():
        for ln in args.out.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                done.add(json.loads(ln)["question"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[续跑] 已有 {len(done)} 条，将跳过")

    todo = [q for q in questions if q not in done]
    if args.limit > 0:
        todo = todo[: args.limit]
    if not todo:
        print("[完成] 没有待采集的问题")
        return 0

    system_prompt = build_system_prompt()
    print(f"[1/3] 待采集 {len(todo)} 条问题（语料共 {len(questions)} 条）")
    print(f"      system 提示词来自：{PROMPT_PATH.name}")

    print("[2/3] 初始化客户端（需要 MySQL / Qdrant / ES / Embedding 均已启动）")
    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    embedding_client_manager.init_client()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = failed = 0
    reason_counter: dict[str, int] = {}

    print("[3/3] 逐条跑流水线并校验")
    try:
        assert dw_mysql_client_manager.session_factory
        assert meta_mysql_client_manager.session_factory
        async with (
            dw_mysql_client_manager.session_factory() as dw_session,
            meta_mysql_client_manager.session_factory() as meta_session,
        ):
            dw_repo = DWMysqlRepository(dw_session)

            with args.out.open("a", encoding="utf-8") as fout, \
                 args.reject.open("a", encoding="utf-8") as frej:
                for idx, q in enumerate(todo, 1):
                    try:
                        rec = await run_one(q, dw_session, meta_session)
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        reason_counter["跑图异常"] = reason_counter.get("跑图异常", 0) + 1
                        frej.write(json.dumps(
                            {"question": q, "stage": "graph", "reason": str(exc)},
                            ensure_ascii=False) + "\n")
                        frej.flush()
                        print(f"  [{idx}/{len(todo)}] 跑图异常: {q} -> {str(exc)[:60]}")
                        continue

                    sql = rec["sql"]
                    # ---- 过滤 ----
                    if not sql:
                        rejected += 1
                        reason = "未生成 SQL"
                    elif rec["error"]:
                        rejected += 1
                        reason = f"校验未通过(已校正{rec['correct_sql_count']}次)"
                    else:
                        ok, why = await check_sql_executable(sql, dw_repo)
                        if ok:
                            reason = ""
                        else:
                            rejected += 1
                            reason = why

                    if reason:
                        reason_counter[reason.split(":")[0]] = reason_counter.get(reason.split(":")[0], 0) + 1
                        frej.write(json.dumps(
                            {"question": q, "stage": "filter", "reason": reason, "sql": sql},
                            ensure_ascii=False) + "\n")
                        frej.flush()
                        if idx % 20 == 0 or idx <= 5:
                            print(f"  [{idx}/{len(todo)}] 丢弃: {reason} | {q[:28]}")
                        continue

                    # ---- 通过：写成 sharegpt 格式 ----
                    sample = {
                        "question": q,
                        "capability_hint": "SAFETY_DIALECT" if _is_safety(q) else (
                            "CLARIFY" if rec["is_clarify_like"] else "SQL_GEN"),
                        "correct_sql_count": rec["correct_sql_count"],
                        "reference_sql": sql,
                        "conversations": [
                            {"from": "system", "value": system_prompt},
                            {"from": "human", "value": build_human_prompt(rec, q)},
                            {"from": "gpt", "value": sql},
                        ],
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fout.flush()
                    kept += 1
                    if idx % 20 == 0:
                        print(f"  [{idx}/{len(todo)}] 已保留 {kept} 条")
    finally:
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()
        await es_client_manager.close()
        await qdrant_client_manager.close()

    print()
    print(f"采集完成：保留 {kept} 条 | 丢弃 {rejected} 条 | 跑图异常 {failed} 条")
    for r, n in sorted(reason_counter.items(), key=lambda kv: -kv[1]):
        print(f"  丢弃原因 [{r}] {n} 条")
    print(f"训练集 -> {args.out}")
    print(f"丢弃记录 -> {args.reject}（用于分析流水线短板）")
    print()
    print("提示：收集到的样本还需要**手工补充澄清类与安全类负样本**，")
    print("      因为当前流水线不具备澄清能力，产不出这类样本（见 finetune/微调规划.md 3.1 节）。")
    return 0


def _is_safety(q: str) -> bool:
    return bool(re.search(r"删|改|更新|插入|DROP|truncate|密码|清理", q, re.IGNORECASE))


def main() -> int:
    parser = argparse.ArgumentParser(description="用现有流水线采集微调训练数据")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reject", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--resume", action="store_true", help="跳过已采集的问题，断点续跑")
    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[中断] 已采集的样本已落盘，可加 --resume 继续", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
