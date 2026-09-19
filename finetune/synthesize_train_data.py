#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速合成 generate_sql 训练数据（替代跑整条流水线采集）。

为什么改用合成
--------------
原来的 build_train_set.py 让每条问题都走完整流水线，而流水线里
5 次大模型调用中有 4 次是"召回/过滤"，只有最后一次才是要训的 generate_sql。
在只有 5 张表的库上，为每条问题重跑一遍"找表"流程纯属浪费 ——
实测 0.42 分钟/条，攒够训练集要 7 小时。

本脚本：schema 直接从 meta 库读一次并复用，每条问题只调 1 次大模型，
并发执行，实测可达每秒数条。产出格式与 build_train_set.py **完全一致**，
两者可以混用。

产出仍是 LLaMA-Factory 的 sharegpt 格式：
    system = generate_sql.prompt 的角色与任务要求
    human  = 【可用数据表信息】+【指标信息】+【时间】+【数据库】+【用户问题】
    gpt    = 标准 SQL（经 EXPLAIN + 实际执行双重校验）

为保证 human 段与线上一致，脚本会按问题内容挑出相关表（含中间桥接表），
而不是每条形都塞全部 5 张表。

用法
----
    python finetune/synthesize_train_data.py --limit 8      # 冒烟
    python finetune/synthesize_train_data.py --target 500   # 补到 500 条
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

import yaml                                                     # noqa: E402
from sqlalchemy import text                                     # noqa: E402

from app.agent.llm import llm                                   # noqa: E402
from app.clients.mysql_client_manager import (                  # noqa: E402
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.prompt.prompt_loader import load_prompt                 # noqa: E402
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository  # noqa: E402

DATA_DIR = HERE / "data"
CORPUS = DATA_DIR / "corpus.txt"
DEFAULT_OUT = DATA_DIR / "train_synth.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompt" / "generate_sql.prompt"

# 事实表始终包含；维表按问题内容挑选
FACT_TABLE = "fact_order"
ALL_TABLES = ["fact_order", "dim_region", "dim_product", "dim_date", "dim_customer"]

# 每张维表对应的"触发词"（命中则把该表纳入 schema）
TABLE_KEYWORDS = {
    "dim_region": ["地区", "大区", "区域", "省份", "省", "华南", "华东", "西南", "华北", "华中",
                   "广东", "浙江", "四川", "北京", "上海", "湖北", "国家"],
    "dim_product": ["商品", "产品", "品类", "品牌", "手机数码", "家用电器", "鞋靴", "服饰",
                    "食品饮料", "休闲零食", "苹果", "华为", "三星", "戴森", "美的", "耐克",
                    "阿迪达斯", "优衣库", "李维斯", "雀巢", "蒙牛", "乐事", "奥利奥",
                    "iPhone", "Galaxy", "Mate", "Kindle", "Instant Pot"],
    "dim_date": ["月", "季度", "年", "日", "时间", "Q1", "第一", "第二", "第三", "趋势", "每天"],
    "dim_customer": ["客户", "用户", "会员", "等级", "性别", "男", "女", "黄金", "白银", "青铜", "铂金",
                     "客户名", "人名"],
}


def build_system_prompt() -> str:
    """与 build_train_set.py 保持完全一致，保证训练/推理格式统一"""
    if not PROMPT_PATH.exists():
        return "你是掌柜问数的 SQL 生成助手。只依据给定的表结构与指标口径生成一条 MySQL 只读查询。"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = re.sub(r"\{[a-z_]+\}", "", text)
    text = re.split(r"用户查询", text)[0]
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


# ==========================================================================
# 从 meta / dw 库读出一次 schema，之后复用
# ==========================================================================
async def load_schema(meta_session, dw_session) -> tuple[list[dict], list[dict], dict]:
    """返回 (table_infos, metric_infos, db_info)，结构与线上 merge_retrieved_info / add_extra_context 的产物一致"""
    # 表信息
    rows = await meta_session.execute(
        text("SELECT id, name, role, description FROM table_info")
    )
    tables = {r.id: {"name": r.name, "role": r.role, "description": r.description, "columns": []}
              for r in rows}

    # 字段信息
    rows = await meta_session.execute(text(
        "SELECT table_id, name, type, role, examples, description, alias FROM column_info"
    ))
    for r in rows:
        if r.table_id not in tables:
            continue
        tables[r.table_id]["columns"].append({
            "name": r.name,
            "type": r.type,
            "role": r.role,
            "examples": list(r.examples or [])[:10],
            "description": r.description,
            "alias": list(r.alias or []),
        })

    # 指标信息
    rows = await meta_session.execute(text(
        "SELECT name, description, relevant_columns, alias FROM metric_info"
    ))
    metrics = [{
        "name": r.name,
        "description": r.description,
        "relevant_columns": list(r.relevant_columns or []),
        "alias": list(r.alias or []),
    } for r in rows]

    # 数据库版本/方言（与线上 add_extra_context 一致）
    ver = (await dw_session.execute(text("select version()"))).scalar()
    db_info = {"version": ver, "dialect": dw_session.bind.dialect.name}

    return list(tables.values()), metrics, db_info


def pick_tables(question: str) -> list[str]:
    """按问题内容挑出相关表；不命中任何维表时给出两张常用表，避免 schema 过窄"""
    picked = [FACT_TABLE]
    for tbl, kws in TABLE_KEYWORDS.items():
        if any(k in question for k in kws):
            picked.append(tbl)
    if len(picked) == 1:
        # 兜底：给事实表 + 最常用的两个维表（金额/销量类问题通常够用）
        picked += ["dim_product", "dim_date"]
    return [t for t in ALL_TABLES if t in set(picked)]


def build_human(table_infos, metric_infos, date_info, db_info, question) -> str:
    def dump(obj) -> str:
        if not obj:
            return "（无）"
        return yaml.dump(obj, allow_unicode=True, sort_keys=False).strip()

    return (
        "【可用数据表信息】\n"
        f"{dump(table_infos)}\n\n"
        "【可参考的指标信息】\n"
        f"{dump(metric_infos)}\n\n"
        "【当前时间信息】\n"
        f"{dump(date_info)}\n\n"
        "【数据库环境】\n"
        f"{dump(db_info)}\n\n"
        "【用户查询】\n"
        f"{question}"
    )


# ==========================================================================
def _has_out_of_range_year(sql: str, data_years: set[int]) -> bool:
    """
    判断 SQL 是否用了数据范围之外的年份过滤。

    背景：语料里的问题大多没提年份，但模型会自己臆造"今年"（当前系统年份 2026），
    生成 WHERE year = 2026。而 dw.dim_date 只覆盖 2025 年，
    这类 SQL 必然返回 NULL —— 是**错误标注**，会把模型带坏，必须剔除。
    """
    for y in re.findall(r"\byear\s*=\s*(\d{4})", sql, re.IGNORECASE):
        if int(y) not in data_years:
            return True
    for y in re.findall(r"date_id\s*>=\s*(\d{8})", sql, re.IGNORECASE):
        if int(y) // 10000 not in data_years:
            return True
    return False


def _rows_are_meaningless(rows: list) -> bool:
    """
    判断结果集是否"形同空集"。

    关键坑：`SELECT SUM(x)` 匹配不到行时返回的是 **一行 [None]**，不是空列表。
    原实现只判断 `if not rows`，因此漏掉了所有聚合查询的失败情况，
    导致 year=2026 这类必然查空的 SQL 被当成有效样本存了下来。
    """
    if not rows:
        return True
    for row in rows:
        vals = row if isinstance(row, (list, tuple)) else list(row.values())
        if all(v is None for v in vals):
            return True
    return False


async def generate_one(question, all_tables, metrics, db_info, date_info, system_prompt):
    """对单个问题生成 SQL（1 次大模型调用）"""
    names = pick_tables(question)
    subset = [t for t in all_tables if t["name"] in names]
    human = build_human(subset, metrics, date_info, db_info, question)
    messages = [("system", system_prompt), ("human", human)]
    resp = await llm.ainvoke(messages)
    sql = (getattr(resp, "content", None) or str(resp)).strip()
    # 去掉可能的 markdown 代码块围栏
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql, human, names


async def main_async(args) -> int:
    if not CORPUS.exists():
        print(f"[错误] 找不到语料 {CORPUS}，请先运行 make_corpus.py", file=sys.stderr)
        return 2
    questions = [ln.strip() for ln in CORPUS.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]

    # 跳过已生成的
    done: set[str] = set()
    if args.resume and args.out.exists():
        for ln in args.out.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    done.add(json.loads(ln)["question"])
                except (json.JSONDecodeError, KeyError):
                    pass
    questions = [q for q in questions if q not in done]
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    if args.target > 0:
        questions = questions[: args.target]
    if not questions:
        print("[完成] 没有待生成的问题")
        return 0

    print(f"[1/3] 待生成 {len(questions)} 条（并发 {args.concurrency}）")

    meta_mysql_client_manager.init_client()
    dw_mysql_client_manager.init_client()
    try:
        assert meta_mysql_client_manager.session_factory
        assert dw_mysql_client_manager.session_factory
        async with (
            meta_mysql_client_manager.session_factory() as meta_session,
            dw_mysql_client_manager.session_factory() as dw_session,
        ):
            from datetime import datetime
            today = datetime.today()
            date_info = {
                "date": today.strftime("%Y-%m-%d"),
                "weekday": today.strftime("%A"),
                "quarter": f"Q{(today.month + 2) // 3}",
            }
            all_tables, metrics, db_info = await load_schema(meta_session, dw_session)
            # 数据实际覆盖的年份，用于剔除"臆造年份"的 SQL
            year_rows = await dw_session.execute(
                text("SELECT DISTINCT year FROM dim_date"))
            data_years = {int(r[0]) for r in year_rows}
            print(f"      schema 载入：{len(all_tables)} 张表 / {len(metrics)} 个指标（复用，不再逐条检索）")
            print(f"      数据实际年份：{sorted(data_years)}（用于剔除超范围年份过滤）")

            system_prompt = build_system_prompt()
            dw_repo = DWMysqlRepository(dw_session)
            sem = asyncio.Semaphore(args.concurrency)
            args.out.parent.mkdir(parents=True, exist_ok=True)

            kept = dropped = failed = 0
            reason_counter: dict[str, int] = {}
            lock = asyncio.Lock()
            fh = args.out.open("a", encoding="utf-8")

            async def worker(q: str):
                nonlocal kept, dropped, failed
                async with sem:
                    try:
                        sql, human, names = await generate_one(
                            q, all_tables, metrics, db_info, date_info, system_prompt)
                    except Exception as exc:  # noqa: BLE001
                        async with lock:
                            failed += 1
                            if failed <= 3:
                                print(f"  [生成异常] {q[:24]} -> {str(exc)[:80]}")
                        return
                    # ---- 双重校验：EXPLAIN + 实际执行 ----
                    if not sql:
                        async with lock:
                            dropped += 1
                        return
                    # 臆造的数据范围外年份过滤 -> 错误标注，直接剔除
                    if _has_out_of_range_year(sql, data_years):
                        async with lock:
                            dropped += 1
                            reason_counter["超范围年份过滤"] = reason_counter.get("超范围年份过滤", 0) + 1
                        return
                    try:
                        await dw_repo.validate_sql(sql)
                    except Exception:  # noqa: BLE001
                        async with lock:
                            dropped += 1
                            reason_counter["EXPLAIN 失败"] = reason_counter.get("EXPLAIN 失败", 0) + 1
                        return
                    try:
                        rows = await dw_repo.execute_sql(sql)
                    except Exception:  # noqa: BLE001
                        async with lock:
                            dropped += 1
                            reason_counter["执行失败"] = reason_counter.get("执行失败", 0) + 1
                        return
                    # 空集 或 全 NULL（聚合查询查不到行的形态）都算无效
                    if _rows_are_meaningless(rows):
                        async with lock:
                            dropped += 1
                            reason_counter["结果为空或全NULL"] = reason_counter.get("结果为空或全NULL", 0) + 1
                        return

                    sample = {
                        "question": q,
                        "source": "synthesized",
                        "tables_used": names,
                        "reference_sql": sql,
                        "conversations": [
                            {"from": "system", "value": system_prompt},
                            {"from": "human", "value": human},
                            {"from": "gpt", "value": sql},
                        ],
                    }
                    async with lock:
                        fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        fh.flush()
                        kept += 1
                        if kept % 25 == 0:
                            print(f"  已保留 {kept} 条（丢弃 {dropped} / 异常 {failed}）")

            print("[2/3] 并发生成中……")
            await asyncio.gather(*(worker(q) for q in questions))
            fh.close()
    finally:
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()

    print()
    print(f"[3/3] 保留 {kept} 条 | 丢弃 {dropped} 条（未通过校验）| 异常 {failed} 条")
    if reason_counter:
        print("  丢弃原因分布：")
        for k, v in sorted(reason_counter.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<20} {v}")
    print(f"输出 -> {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="快速合成 generate_sql 训练数据")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=0, help="生成多少条，0=语料全部")
    p.add_argument("--limit", type=int, default=0, help="同 --target，冒烟用")
    p.add_argument("--concurrency", type=int, default=8, help="并发大模型请求数")
    p.add_argument("--seed", type=int, default=20250923)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.limit and not args.target:
        args.target = args.limit
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[中断] 已生成样本已落盘，可加 --resume 继续", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
