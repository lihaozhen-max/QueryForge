#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享工具：由 dw.sql 构建「影子数仓」（SQLite）。

为什么需要它
------------
标准 SQL 若写错表名/列名，肉眼很难发现，但评测时会全部判错，
微调效果的对比就失去意义。因此需要一个**零成本、可离线**的执行环境：

    validate_sql.py   用它校验生成的标准 SQL 是否可执行
    eval_ex.py        用它做执行准确率（EX）的离线评测

两个脚本共用本模块，避免重复实现。

局限（务必知悉）
----------------
这是 MySQL -> SQLite 的**近似**环境，只能验证表名/列名/语法结构与结果集。
MySQL 特有的方言（LIMIT 之外的 TOP、DATE_FORMAT、反引号等）不在覆盖范围内。
因此：
    * 离线跑通 ≠ 线上一定通过 —— 最终评测请连真实 MySQL（见 eval_ex.py --db mysql）
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# 教师工程里的建表 + 初始化数据脚本
DEFAULT_SCHEMA_CANDIDATES = [
    Path(r"C:\Users\13273\Desktop\0318掌柜问数\3.代码\docker\mysql\dw.sql"),
]

TABLES = ("dim_region", "dim_customer", "dim_product", "dim_date", "fact_order")

# MySQL 专有语句：影子库不需要执行
_SKIP = re.compile(
    r"^\s*(CREATE\s+DATABASE|DROP\s+DATABASE|GRANT|REVOKE|USE|SET\s+NAMES|FLUSH|LOCK|UNLOCK)",
    re.IGNORECASE,
)


def find_schema(explicit: Path | str | None = None, extra_dirs: list[Path] | None = None) -> Path | None:
    """定位 dw.sql：优先用显式路径，其次查默认位置与额外目录。"""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for cand in DEFAULT_SCHEMA_CANDIDATES:
        if cand.exists():
            return cand
    for d in extra_dirs or []:
        p = d / "dw.sql"
        if p.exists():
            return p
    return None


def build_shadow_db(schema_path: Path, quiet: bool = False) -> sqlite3.Connection:
    """用 dw.sql 的 DDL + INSERT 建一个 SQLite 内存影子库。"""
    sql_text = Path(schema_path).read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")

    executed = 0
    for stmt in (s.strip() for s in sql_text.split(";") if s.strip()):
        if _SKIP.match(stmt):
            continue
        try:
            conn.execute(stmt)
            executed += 1
        except sqlite3.Error as exc:
            if not quiet:
                print(f"  [警告] 建库语句跳过: {exc}")
    conn.commit()
    if not quiet:
        print(f"      影子库执行 {executed} 条语句")
    return conn


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in TABLES:
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.Error:
            out[t] = -1
    return out


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    """
    把结果集规范化，用于「结果集等价」比较。

    处理三件事：
      1. 数字统一成 float 再规整，避免 8999 与 8999.0 被判为不同；
      2. 字符串 strip；
      3. 行序无关 —— 排序后比较（同一业务问题允许多条等价 SQL，
         也可能因 ORDER BY 差异导致行序不同，行序不应影响正确性）。
    """
    def norm_cell(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return round(float(v), 6)
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        s = str(v).strip()
        # 纯数字字符串也按数字比
        try:
            return round(float(s), 6)
        except ValueError:
            return s

    normalized = [tuple(norm_cell(c) for c in row) for row in rows]
    return sorted(normalized, key=lambda r: tuple((c is None, str(c)) for c in r))


def rows_equal(a: list[tuple], b: list[tuple]) -> bool:
    """判断两个结果集是否等价（行序无关、数字按数值比）。"""
    return normalize_rows(a) == normalize_rows(b)


def run_sql(conn: sqlite3.Connection, sql: str, limit: int = 10000) -> tuple[bool, list[tuple], str]:
    """
    在影子库执行一条 SQL。
    返回 (是否成功, 结果行, 错误信息)。
    """
    if not sql or not sql.strip():
        return False, [], "空 SQL"
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return False, [], "非只读查询语句"
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(limit)
        return True, rows, ""
    except sqlite3.Error as exc:
        return False, [], str(exc)
