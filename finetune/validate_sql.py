#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校验评测集里的标准 SQL 是否真的可执行。

做法
----
用 SQLite 内存库 + dw.sql 里的真实 DDL/数据建一个「影子数仓」，
把 test.jsonl 中每条 SQL 类样本的 reference_sql 跑一遍，确认能执行出结果。

为什么这样做
    生成的 SQL 若写错表名/列名，肉眼很难发现，但评测时会全部判错，
    导致微调效果的对比毫无意义。先在影子库上跑通，是零成本的兜底。

局限（务必知悉）
    这是 MySQL -> SQLite 的近似校验，只能验证「表名/列名/语法结构」正确，
    不能覆盖 MySQL 特有函数与方言。因此：
      * 本脚本用于**离线快速回归**，保证 SQL 结构无误；
      * 最终执行准确率（EX）评测必须连**真实 MySQL**，见 eval_ex.py。

用法
----
    python finetune/validate_sql.py
"""

from __future__ import annotations

import json
import re
import sqlite3
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
TEST_SET = HERE / "data" / "test.jsonl"
DEFAULT_SCHEMA_CANDIDATES = [
    Path(r"C:\Users\13273\Desktop\0318掌柜问数\3.代码\docker\mysql\dw.sql"),
    HERE / "dw.sql",
]


def build_shadow_db(schema_path: Path) -> sqlite3.Connection:
    """用 dw.sql 的 DDL + INSERT 建一个 SQLite 影子库。"""
    sql_text = schema_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(":memory:")
    # dw.sql 刻意不含外键、不含反引号，因此 DDL 基本可直接执行；
    # 仅需去掉 MySQL 专有的库级语句。
    skip = re.compile(r"^\s*(CREATE\s+DATABASE|GRANT|USE|SET\s+NAMES|DROP\s+DATABASE)", re.IGNORECASE)

    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    executed = 0
    for stmt in statements:
        if skip.match(stmt):
            continue
        # GRANT/USE 等可能被包在多行里，二次剔除
        if re.search(r"\b(GRANT|CREATE\s+DATABASE)\b", stmt, re.IGNORECASE):
            continue
        try:
            conn.execute(stmt)
            executed += 1
        except sqlite3.Error as exc:
            print(f"  [警告] 建库语句跳过: {exc}\n         {stmt[:90]}...")
    conn.commit()
    print(f"      影子库执行 {executed} 条语句")
    return conn


def main() -> int:
    schema_path = next((p for p in DEFAULT_SCHEMA_CANDIDATES if p.exists()), None)
    if schema_path is None:
        print("[错误] 找不到 dw.sql，无法建影子库", file=sys.stderr)
        return 2
    if not TEST_SET.exists():
        print(f"[错误] 找不到评测集 {TEST_SET}，请先运行 gen_eval_set.py", file=sys.stderr)
        return 2

    print(f"[1/3] 用 {schema_path.name} 构建 SQLite 影子库")
    conn = build_shadow_db(schema_path)

    print("[2/3] 检查表数据量")
    for tbl in ("dim_region", "dim_customer", "dim_product", "dim_date", "fact_order"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"      {tbl:<14} {n:>5} 行")

    print("[3/3] 逐条执行标准 SQL")
    records = [json.loads(line) for line in TEST_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    ok, empty, failed, skipped = 0, 0, [], 0
    for rec in records:
        sql = rec.get("reference_sql")
        if rec["expected_type"] != "SQL" or not sql:
            skipped += 1
            continue
        # 仅允许只读查询
        if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
            failed.append((rec["id"], "非 SELECT 语句", sql))
            continue
        try:
            rows = conn.execute(sql).fetchall()
            if rows:
                ok += 1
            else:
                empty += 1
        except sqlite3.Error as exc:
            failed.append((rec["id"], str(exc), sql))

    total_sql = ok + empty + len(failed)
    print()
    print(f"      SQL 类样本共 {total_sql} 条（跳过非 SQL 类 {skipped} 条）")
    print(f"      可执行且有结果 : {ok}")
    print(f"      可执行但结果为空: {empty}")
    print(f"      执行失败        : {len(failed)}")
    print()

    if empty:
        print("  [提示] 结果为空不一定是错误，但应确认数据里确实没有匹配行：")
        for rec in records:
            sql = rec.get("reference_sql")
            if rec["expected_type"] != "SQL" or not sql:
                continue
            try:
                if not conn.execute(sql).fetchall():
                    print(f"        - [{rec['id']}] {rec['question']}")
            except sqlite3.Error:
                pass
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
