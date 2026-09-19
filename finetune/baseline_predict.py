#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跑现状基线（P1）：用当前 LangGraph 流水线 + 现有大模型，对评测集生成预测。

为什么需要
----------
微调要有"前后对比"，就必须先有**微调前的基准指标**。本脚本把现状流水线
在 test.jsonl 上的输出落成预测文件，交给 eval_ex.py 算三层指标，
填进 微调规划.md 的对比矩阵第一行。

为什么不用 HTTP 接口
--------------------
POST /api/query 的 SSE 只输出 {stage} 与 {result}，拿不到生成的 SQL，
而执行准确率（EX）必须比对 SQL 的执行结果集。因此这里直接驱动 compiled_graph，
用 stream_mode="values" 取终态（与 build_train_set.py 同一套做法）。

用法
----
    # 4 条冒烟，确认链路通
    python finetune/baseline_predict.py --limit 4

    # 全量 115 条（LLM 调用较慢，建议后台跑）
    python finetune/baseline_predict.py --out finetune/data/preds/baseline_deepseek_flash.jsonl

    # 然后算指标
    python finetune/eval_ex.py --db mysql --predictions <预测文件> --name "基线：现有流水线"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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
DEFAULT_OUT = HERE / "data" / "preds" / "baseline.jsonl"


async def predict_one(question: str, dw_session, meta_session) -> str:
    """跑一次流水线，返回生成的 SQL（拿不到则返回空串）"""
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
    async for snapshot in compiled_graph.astream(
        input=state, context=context, stream_mode="values"
    ):
        if isinstance(snapshot, dict):
            final = snapshot
    return (final.get("sql") or "").strip()


async def main_async(args) -> int:
    records = [
        json.loads(line)
        for line in args.test_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        records = records[: args.limit]

    # 断点续跑
    done: dict[str, dict] = {}
    if args.resume and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[续跑] 已有 {len(done)} 条")

    todo = [r for r in records if r["id"] not in done]
    if not todo:
        print("[完成] 没有待跑样本")
        return 0
    print(f"[1/2] 待跑 {len(todo)} 条（评测集共 {len(records)} 条）")

    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    embedding_client_manager.init_client()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        assert dw_mysql_client_manager.session_factory
        assert meta_mysql_client_manager.session_factory
        async with (
            dw_mysql_client_manager.session_factory() as dw_session,
            meta_mysql_client_manager.session_factory() as meta_session,
        ):
            with args.out.open("a", encoding="utf-8") as fh:
                for i, rec in enumerate(todo, 1):
                    # 危险请求：期望模型拒绝；若流水线没这能力，它会尝试生成 SQL，
                    # 原样记录即可，由 eval_ex.py 判定为失败。
                    try:
                        sql = await predict_one(rec["question"], dw_session, meta_session)
                        err = ""
                    except Exception as exc:  # noqa: BLE001
                        sql, err = "", str(exc)[:300]

                    out = {
                        "id": rec["id"],
                        "question": rec["question"],
                        "capability": rec.get("capability"),
                        "expected_type": rec.get("expected_type"),
                        "refuse": rec.get("refuse", False),
                        # 现状流水线只会输出 SQL；没有澄清能力，
                        # 因此 type 固定为 SQL（若为空则由 eval_ex 判为失败）
                        "type": "SQL" if sql else "CLARIFY",
                        "content": sql,
                        "error": err,
                    }
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fh.flush()

                    mark = "OK " if sql else "空 "
                    if i % 5 == 0 or i <= 3 or not sql:
                        el = time.time() - t0
                        print(f"  [{i}/{len(todo)}] {mark}{rec['question'][:30]}  "
                              f"({el:.0f}s, {el/i:.1f}s/条)")
    finally:
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()
        await es_client_manager.close()
        await qdrant_client_manager.close()

    total = len(done) + len(todo)
    el = time.time() - t0
    print()
    print(f"[2/2] 完成：本次 {len(todo)} 条，累计 {total} 条，用时 {el:.0f}s")
    print(f"预测文件 -> {args.out}")
    print()
    print("下一步（算指标，需要 MySQL 可连）：")
    print(f"  python finetune/eval_ex.py --db mysql --predictions {args.out} "
          f"--name \"基线：现有流水线\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="跑现状基线预测（P1）")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    parser.add_argument("--resume", action="store_true", help="跳过已跑过的样本")
    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[中断] 已跑样本已落盘，可加 --resume 继续", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
