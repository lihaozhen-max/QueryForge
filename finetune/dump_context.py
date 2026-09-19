#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出评测集的「检索上下文」，供租卡机器上的模型做预测。

为什么要这一步
--------------
预测 = 「schema 上下文 + 问题」→ 模型生成 SQL。
其中 schema 上下文来自本机的检索链路（Embedding + Qdrant + ES + MySQL），
但**租卡机器上没有这些服务**，无法现场检索。

所以本机先跑一次检索，把 115 条问题各自的 schema 落成文件，
上传到租卡机器后即可脱离检索服务直接生成 SQL。

--fast：为什么默认跳过 LLM 过滤节点
-----------------------------------
流水线里 filter_table / filter_metric 会再调两次大模型，把召回的表和字段裁掉一部分。
它有两个问题：

1. **慢**：每条要 ~100 秒，115 条要 3 小时以上。
2. **引入了第二个变量**：过滤有时会裁掉 JOIN 需要的维表，
   于是模型答错的原因可能是"过滤裁错了"而不是"模型能力不够"。
   这样 base 与微调模型的对比就不纯净。

--fast 直接使用召回结果（不做 LLM 过滤），于是：
* 每条只需 3 次检索、0 次大模型调用 → 全量只要几分钟；
* base 与微调模型使用**完全相同的上下文**，唯一变量只剩模型本身；
* 与训练数据的给表方式也更接近（训练时是按问题挑相关表）。

代价：schema 略长、字段略多，绝对 EX 可能比线上真实表现略低。
但该影响对 base 与微调模型是对称的，不影响"谁更好"的结论。

用法
----
    python finetune/dump_context.py            # 默认 fast，几分钟
    python finetune/dump_context.py --no-fast  # 走完整流水线含 LLM 过滤（慢，数小时）
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from app.repositories.es.value_es_repository import ValueESRepository      # noqa: E402
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository   # noqa: E402
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository  # noqa: E402
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository  # noqa: E402
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository  # noqa: E402

TEST_SET = HERE / "data" / "test.jsonl"
DEFAULT_OUT = HERE / "data" / "eval_context.jsonl"


class _FakeRuntime:
    """给节点函数用的最小 Runtime：只需要 stream_writer 与 context"""

    def __init__(self, context: dict):
        self.context = context

    @staticmethod
    def stream_writer(_payload) -> None:  # 节点会往里写进度，这里忽略
        pass


async def one_fast(question: str, dw_session, meta_session) -> dict:
    """
    快速路径：只跑「提关键词 → 三路召回 → 合并」，跳过两个 LLM 过滤节点。
    全程只有 3 次 embedding + 向量/全文检索，0 次大模型调用。
    """
    from app.agent.nodes.extract_keywords import extract_keywords
    from app.agent.nodes.merge_retrieved_info import merge_retrieved_info
    from app.agent.nodes.recall_column import recall_column
    from app.agent.nodes.recall_metric import recall_metric
    from app.agent.nodes.recall_value import recall_value

    context = DataAgentContext(
        dw_mysql_repo=DWMysqlRepository(dw_session),
        meta_mysql_repo=MetaMysqlRepository(meta_session),
        value_es_repo=ValueESRepository(es_client_manager.client),
        column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
        metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
        embedding_client=embedding_client_manager.client,
    )
    rt = _FakeRuntime(context)
    state: dict = {"query": question}

    # 提关键词（jieba，不调大模型）
    state.update(await extract_keywords(state, rt) or {})
    # 三路召回（各自会调 1 次大模型扩词 —— 这是流水线固有设计，保留）
    for node in (recall_column, recall_metric, recall_value):
        state.update(await node(state, rt) or {})
    # 合并成 table_infos / metric_infos
    state.update(await merge_retrieved_info(state, rt) or {})

    from datetime import datetime
    today = datetime.today()
    return {
        "table_infos": state.get("table_infos") or [],
        "metric_infos": state.get("metric_infos") or [],
        "date_info": {
            "date": today.strftime("%Y-%m-%d"),
            "weekday": today.strftime("%A"),
            "quarter": f"Q{(today.month + 2) // 3}",
        },
        "db_info": {"version": "8.0.46", "dialect": "mysql"},
    }


async def one_full(question: str, dw_session, meta_session) -> dict:
    """完整路径：走整条 LangGraph（含两个 LLM 过滤节点），慢但不改变线上行为"""
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
    async for snap in compiled_graph.astream(input=state, context=context, stream_mode="values"):
        if isinstance(snap, dict):
            final = snap
    return {
        "table_infos": final.get("table_infos") or [],
        "metric_infos": final.get("metric_infos") or [],
        "date_info": final.get("date_info") or {},
        "db_info": final.get("db_info") or {},
    }


async def main_async(args) -> int:
    records = [
        json.loads(line)
        for line in args.test_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.no_fast:
        print(f"[1/2] 评测集 {len(records)} 条；**完整流水线**（含 LLM 过滤，会很慢）")
    else:
        print(f"[1/2] 评测集 {len(records)} 条；**快速模式**（跳过 LLM 过滤节点）")
    runner = one_full if args.no_fast else one_fast

    # 断点续跑：跳过已导出的题目（快速模式下每条约 18s，中断后不必重跑）
    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"      [续跑] 已有 {len(done_ids)} 条，跳过")
    todo = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print("[完成] 没有待导出的样本")
        return 0

    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    embedding_client_manager.init_client()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    try:
        assert dw_mysql_client_manager.session_factory
        assert meta_mysql_client_manager.session_factory
        async with (
            dw_mysql_client_manager.session_factory() as dw_session,
            meta_mysql_client_manager.session_factory() as meta_session,
        ):
            with args.out.open("a", encoding="utf-8") as fh:
                for i, rec in enumerate(todo, 1):
                    try:
                        ctx = await runner(rec["question"], dw_session, meta_session)
                    except Exception as exc:  # noqa: BLE001
                        ctx = {"table_infos": [], "metric_infos": [], "date_info": {}, "db_info": {}}
                        print(f"  [{i}] 召回失败: {rec['question'][:24]} -> {str(exc)[:70]}")
                    fh.write(json.dumps({
                        "id": rec["id"],
                        "question": rec["question"],
                        "capability": rec.get("capability"),
                        "expected_type": rec.get("expected_type"),
                        "refuse": rec.get("refuse", False),
                        "missing": rec.get("missing"),
                        "reference_sql": rec.get("reference_sql"),
                        **ctx,
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    ok += 1
                    if i % 20 == 0:
                        print(f"  [{i}/{len(records)}] 已导出 {ok} 条")
    finally:
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()
        await es_client_manager.close()
        await qdrant_client_manager.close()

    print()
    print(f"[2/2] 导出完成：{ok} 条 -> {args.out}")
    print("      上传这个文件到租卡机器，即可脱离检索服务做预测。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="导出评测集的检索上下文")
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-fast", action="store_true",
                    help="走完整流水线（含 LLM 过滤节点，慢，数小时）")
    ap.add_argument("--resume", action="store_true", help="跳过已导出的题目")
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[中断]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
