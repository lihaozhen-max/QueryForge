#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校验评测集里的标准 SQL 是否真的可执行。

做法
----
用 SQLite 内存库 + dw.sql 里的真实 DDL/数据建一个「影子数仓」（见 shadow_db.py），
把 test.jsonl 中每条 SQL 类样本的 reference_sql 跑一遍，确认能执行出结果。

为什么这样做
    生成的 SQL 若写错表名/列名，肉眼很难发现，但评测时会全部判错，
    导致微调效果的对比毫无意义。先在影子库上跑通，是零成本的兜底。

局限（务必知悉）
    这是 MySQL -> SQLite 的近似校验，只能验证「表名/列名/语法结构」正确，
    不能覆盖 MySQL 特有函数与方言。因此：
      * 本脚本用于**离线快速回归**，保证 SQL 结构无误；
      * 最终执行准确率（EX）评测请连**真实 MySQL**，见 eval_ex.py --db mysql。

用法
----
    python finetune/validate_sql.py
    python finetune/validate_sql.py --schema path/to/dw.sql
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认是 GBK，直接 print 中文/符号会抛 UnicodeEncodeError，
# 也会让日志变成乱码。这里统一把标准输出切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from shadow_db import (  # noqa: E402
    TABLES,
    build_shadow_db,
    find_schema,
    run_sql,
    table_counts,
)

TEST_SET = HERE / "data" / "test.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="校验评测集标准 SQL 是否可执行")
    parser.add_argument("--schema", type=Path, default=None, help="dw.sql 路径")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    args = parser.parse_args()

    schema_path = find_schema(args.schema, extra_dirs=[HERE])
    if schema_path is None:
        print("[错误] 找不到 dw.sql，无法建影子库（可用 --schema 指定）", file=sys.stderr)
        return 2
    if not args.test_set.exists():
        print(f"[错误] 找不到评测集 {args.test_set}，请先运行 gen_eval_set.py", file=sys.stderr)
        return 2

    print(f"[1/3] 用 {schema_path.name} 构建 SQLite 影子库")
    conn = build_shadow_db(schema_path)

    print("[2/3] 检查表数据量")
    for tbl, n in table_counts(conn).items():
        print(f"      {tbl:<14} {n:>5} 行")

    print("[3/3] 逐条执行标准 SQL")
    records = [
        json.loads(line)
        for line in args.test_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    ok, empty, failed, skipped = 0, 0, [], 0
    empty_items: list[tuple[str, str]] = []
    for rec in records:
        sql = rec.get("reference_sql")
        if rec.get("expected_type") != "SQL" or not sql:
            skipped += 1
            continue
        good, rows, err = run_sql(conn, sql)
        if not good:
            failed.append((rec["id"], err, sql))
        elif rows:
            ok += 1
        else:
            empty += 1
            empty_items.append((rec["id"], rec["question"]))

    total_sql = ok + empty + len(failed)
    print()
    print(f"      SQL 类样本共 {total_sql} 条（跳过非 SQL 类 {skipped} 条）")
    print(f"      可执行且有结果 : {ok}")
    print(f"      可执行但结果为空: {empty}")
    print(f"      执行失败        : {len(failed)}")
    print()

    if empty_items:
        print("  [提示] 结果为空不一定是错误，但应确认数据里确实没有匹配行：")
        for rid, q in empty_items:
            print(f"        - [{rid}] {q}")
        print()

    if failed:
        print("  [失败明细]")
        for rid, err, sql in failed:
            print(f"        - [{rid}] {err}")
            print(f"          {sql}")
        conn.close()
        return 1

    print("  全部标准 SQL 均可执行 ✅")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
