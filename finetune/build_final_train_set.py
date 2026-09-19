#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并三路数据源，生成最终训练集（LLaMA-Factory sharegpt 格式）。

数据源
------
1. train_synth.jsonl   394 条  问数 → SQL（合成，已过 EXPLAIN + 执行校验）
2. correct_sql.jsonl   541 条  (错误SQL + 真实报错) → 修正 SQL
3. negatives.jsonl      33 条  澄清 / 拒答
   train.jsonl          43 条  早期流水线采集（可选，格式同 1）

本脚本要修的两个关键问题
------------------------
**问题1：system 提示词不一致，模型会学乱。**
    train_synth 用原版 generate_sql.prompt（只输出 SQL），
    negatives 用追加了「信息不足要反问」的版本。
    二者行为相反却共用一段 system，模型无法分辨该给 SQL 还是该反问。
    处理：全部统一为**含澄清约定**的那一版（这也正是 P6 要接入主流程的行为）。

**问题2：negatives 的 human 段没有 schema，澄清能力会学歪。**
    原 negatives 的输入只有「【用户查询】看看销售情况」，
    模型会学到「没有 schema → 就反问」；而线上请求永远带 schema，
    于是它永远不反问。必须补上与其它样本一致的完整上下文。

用法
----
    python finetune/build_final_train_set.py
    python finetune/build_final_train_set.py --val-ratio 0.08
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --------------------------------------------------------------------------
# 统一后的 system 提示词（含澄清约定）——与线上 generate_sql.prompt 保持一致，
# 训练/推理同分布。
# --------------------------------------------------------------------------
UNIFIED_SYSTEM = """【角色】
你是一个资深的数据库专家和数据分析师。你的任务是根据提供的【上下文信息】，将用户的自然语言查询转换为语法正确、性能优化的 SQL 语句。
【上下文信息】
可用数据表信息如下：
可参考的指标信息如下：
当前的时间信息如下：
数据库环境如下：
【任务要求】
1. 仅允许使用数据表信息中真实存在的表与字段名称，禁止编造、猜测或引入未提供的表和字段。
2. 若指标信息中存在相关指标定义，必须严格遵循其业务口径、计算逻辑、过滤规则与时间口径，不得偏离或重解释，若指标信息未覆盖用户问题中的指标或口径，则基于用户问题语义与通用业务常识进行。
3. 生成的SQL只能用于查询，不能涉及数据写入、更新、删除等操作。
4. 生成的SQL的语法必须严格符合数据库环境中指定的数据库类型与版本。
5. 默认只生成一条SQL，不可生成多条SQL。
6. 输出必须仅包含一条完整 SQL 语句的纯文本，严禁使用```、```sql 等 Markdown代码块或任何格式化符号。

【输出类型】
你必须二选一输出：
A. 若问题已具备「时间范围」「统计粒度」「指标口径」等必要信息，输出一条 MySQL 只读查询语句；
B. 若信息不足，不要臆测，改为提出澄清问题：先用一句话指出缺少哪些信息，再列出 2-3 个具体可选的确认项，每个确认项给出可选值，便于用户直接回答。
无论哪种情况，都只输出纯文本，不要使用 Markdown 代码块。"""

# 维表触发词（与 synthesize_train_data.py 的 pick_tables 保持一致）
TABLE_KEYWORDS = {
    "dim_region": ["地区", "大区", "区域", "省份", "省", "华南", "华东", "西南", "华北", "华中",
                   "广东", "浙江", "四川", "北京", "上海", "湖北", "国家", "渠道"],
    "dim_product": ["商品", "产品", "品类", "品牌", "手机数码", "家用电器", "鞋靴", "服饰",
                    "食品饮料", "休闲零食", "苹果", "华为", "三星", "戴森", "美的", "耐克",
                    "阿迪达斯", "优衣库", "李维斯", "雀巢", "蒙牛", "乐事", "奥利奥",
                    "iPhone", "Galaxy", "Mate", "Kindle", "Instant Pot"],
    "dim_date": ["月", "季度", "年", "日", "时间", "Q1", "第一", "第二", "第三", "趋势", "每天",
                 "同比增长", "环比", "最近"],
    "dim_customer": ["客户", "用户", "会员", "等级", "性别", "男", "女", "黄金", "白银", "青铜", "铂金"],
}
ALL_TABLES = ["fact_order", "dim_region", "dim_product", "dim_date", "dim_customer"]

# 通用兜底 schema 段（无法从 train_synth 借到 schema 时使用）
FALLBACK_TABLES = [
    {"name": "fact_order", "role": "fact",
     "description": "订单事实表，记录订单数量和金额等核心指标。",
     "columns": [
         {"name": "order_id", "type": "varchar(30)", "role": "primary_key",
          "examples": ["ORD20250101001"], "description": "订单唯一标识。", "alias": ["订单ID"]},
         {"name": "customer_id", "type": "varchar(20)", "role": "foreign_key",
          "examples": ["C001"], "description": "关联客户维度的外键。", "alias": ["客户ID", "用户ID"]},
         {"name": "product_id", "type": "varchar(20)", "role": "foreign_key",
          "examples": ["P001"], "description": "关联商品维度的外键。", "alias": ["商品ID", "产品ID"]},
         {"name": "date_id", "type": "int", "role": "foreign_key",
          "examples": [20250101], "description": "关联时间维度的外键。", "alias": ["日期", "下单日期"]},
         {"name": "region_id", "type": "varchar(20)", "role": "foreign_key",
          "examples": ["R001"], "description": "关联地区维度的外键。", "alias": ["地区ID", "区域ID"]},
         {"name": "order_quantity", "type": "int", "role": "measure",
          "examples": [1, 5], "description": "订单中商品的购买数量。", "alias": ["销量", "购买数量", "件数"]},
         {"name": "order_amount", "type": "float", "role": "measure",
          "examples": [8999.0], "description": "订单金额。", "alias": ["销售额", "订单金额", "收入"]},
     ]},
    {"name": "dim_region", "role": "dim", "description": "地区维度表，用于描述订单发生的地理区域信息。",
     "columns": [
         {"name": "region_id", "type": "varchar(20)", "role": "primary_key",
          "examples": ["R001"], "description": "地区唯一标识。", "alias": ["地区ID", "区域ID"]},
         {"name": "province", "type": "varchar(50)", "role": "dimension",
          "examples": ["广东省"], "description": "订单所属的省份名称。", "alias": ["省份", "省"]},
         {"name": "region_name", "type": "varchar(50)", "role": "dimension",
          "examples": ["华南"], "description": "订单所属的大区名称。", "alias": ["地区", "区域", "大区"]},
     ]},
    {"name": "dim_product", "role": "dim", "description": "商品维度表，描述商品的基本属性信息。",
     "columns": [
         {"name": "product_id", "type": "varchar(20)", "role": "primary_key",
          "examples": ["P001"], "description": "商品唯一标识。", "alias": ["商品ID", "产品ID"]},
         {"name": "product_name", "type": "varchar(200)", "role": "dimension",
          "examples": ["iPhone 15 Pro"], "description": "商品名称。", "alias": ["商品名称", "产品名称"]},
         {"name": "category", "type": "varchar(50)", "role": "dimension",
          "examples": ["手机数码"], "description": "商品所属品类。", "alias": ["商品类别", "品类", "分类"]},
         {"name": "brand", "type": "varchar(50)", "role": "dimension",
          "examples": ["苹果"], "description": "商品品牌名称。", "alias": ["品牌", "品牌名称"]},
     ]},
    {"name": "dim_date", "role": "dim", "description": "时间维度表，用于多时间粒度分析。",
     "columns": [
         {"name": "date_id", "type": "int", "role": "primary_key",
          "examples": [20250101], "description": "日期唯一标识，格式 yyyyMMdd。", "alias": ["日期ID", "日期"]},
         {"name": "year", "type": "int", "role": "dimension",
          "examples": [2025], "description": "年份。", "alias": ["年", "年份"]},
         {"name": "quarter", "type": "varchar(2)", "role": "dimension",
          "examples": ["Q1"], "description": "季度。", "alias": ["季度"]},
         {"name": "month", "type": "int", "role": "dimension",
          "examples": [1], "description": "月份。", "alias": ["月", "月份"]},
     ]},
    {"name": "dim_customer", "role": "dim", "description": "客户维度表，描述下单客户的基本属性。",
     "columns": [
         {"name": "customer_id", "type": "varchar(20)", "role": "primary_key",
          "examples": ["C001"], "description": "客户唯一标识。", "alias": ["客户ID", "用户ID"]},
         {"name": "customer_name", "type": "varchar(50)", "role": "dimension",
          "examples": ["李伟"], "description": "客户名称。", "alias": ["客户名称", "用户名称"]},
         {"name": "gender", "type": "varchar(10)", "role": "dimension",
          "examples": ["男"], "description": "客户性别。", "alias": ["性别"]},
         {"name": "member_level", "type": "varchar(20)", "role": "dimension",
          "examples": ["黄金"], "description": "客户会员等级。", "alias": ["会员等级", "用户等级"]},
     ]},
]

FALLBACK_METRICS = [
    {"name": "GMV", "description": "全称Gross Merchandise Value，表示所有订单的成交金额总和。",
     "relevant_columns": ["fact_order.order_amount"], "alias": ["成交总额", "订单总额"]},
    {"name": "AOV", "description": "全称Average Order Value，表示所有订单的成交金额平均值。",
     "relevant_columns": ["fact_order.order_amount", "fact_order.order_quantity"],
     "alias": ["平均单价", "平均订单金额"]},
]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def pick_tables(question: str) -> list[str]:
    picked = ["fact_order"]
    for tbl, kws in TABLE_KEYWORDS.items():
        if any(k in question for k in kws):
            picked.append(tbl)
    if len(picked) == 1:
        picked += ["dim_product", "dim_date"]
    return [t for t in ALL_TABLES if t in set(picked)]


def extract_templates(sources: list[list[dict]]) -> tuple[dict, dict]:
    """
    从已有样本里抽取「表结构模板」与「指标信息」的 yaml 文本，用于给 negatives 补上下文。
    这样保证补出来的 schema 与真实检索结果同源，而不是另写一份。
    """
    import yaml
    table_tpl: dict[str, dict] = {}
    metrics_tpl: list[dict] = []

    for recs in sources:
        for r in recs:
            conv = r.get("conversations") or []
            human = conv[1]["value"] if len(conv) > 1 else ""
            # 解析【可用数据表信息】段
            if "【可用数据表信息】" in human:
                seg = human.split("【可用数据表信息】")[1].split("【")[0]
                try:
                    data = yaml.safe_load(seg)
                    if isinstance(data, list):
                        for t in data:
                            if isinstance(t, dict) and t.get("name"):
                                # 保留信息最全的一份
                                old = table_tpl.get(t["name"])
                                if old is None or len(t.get("columns") or []) > len(old.get("columns") or []):
                                    table_tpl[t["name"]] = t
                except Exception:  # noqa: BLE001
                    pass
            if "【可参考的指标信息】" in human and not metrics_tpl:
                seg = human.split("【可参考的指标信息】")[1].split("【")[0]
                try:
                    data = yaml.safe_load(seg)
                    if isinstance(data, list):
                        metrics_tpl = data
                except Exception:  # noqa: BLE001
                    pass
    return table_tpl, metrics_tpl


def build_human(question: str, table_tpl: dict, metrics: list) -> str:
    """按问题挑选相关表，拼出与线上一致的 human 段"""
    import yaml
    names = pick_tables(question)
    tables = [table_tpl[n] for n in names if n in table_tpl]
    if not tables:
        tables = FALLBACK_TABLES

    def dump(o) -> str:
        return yaml.dump(o, allow_unicode=True, sort_keys=False).strip() if o else "（无）"

    date_info = {"date": "2026-09-19", "weekday": "Saturday", "quarter": "Q3"}
    db_info = {"version": "8.0.46", "dialect": "mysql"}
    return (
        "【可用数据表信息】\n" + dump(tables) + "\n\n"
        "【可参考的指标信息】\n" + dump(metrics or FALLBACK_METRICS) + "\n\n"
        "【当前时间信息】\n" + dump(date_info) + "\n\n"
        "【数据库环境】\n" + dump(db_info) + "\n\n"
        "【用户查询】\n" + question
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="合并生成最终训练集")
    ap.add_argument("--val-ratio", type=float, default=0.08, help="验证集比例")
    ap.add_argument("--correct-ratio", type=float, default=0.5,
                    help="correct_sql 相对 generate_sql 的目标比例（0=不降采样）")
    ap.add_argument("--clarify-ratio", type=float, default=0.18,
                    help="clarify 相对 generate_sql 的目标比例（0=不降采样）")
    ap.add_argument("--seed", type=int, default=20250924)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    src_sql = load_jsonl(DATA / "train_synth.jsonl")
    src_pipe = load_jsonl(DATA / "train.jsonl")
    src_correct = load_jsonl(DATA / "correct_sql.jsonl")
    src_neg = load_jsonl(DATA / "negatives.jsonl")
    print(f"[1/4] 载入数据源")
    print(f"      train_synth   {len(src_sql):>4} 条（问数→SQL，合成）")
    print(f"      train(流水线)  {len(src_pipe):>4} 条（问数→SQL）")
    print(f"      correct_sql   {len(src_correct):>4} 条（纠错）")
    print(f"      negatives     {len(src_neg):>4} 条（澄清/拒答）")

    # ---- 抽取 schema 模板，用于给 negatives 补上下文 ----
    table_tpl, metrics_tpl = extract_templates([src_sql, src_pipe])
    print(f"      抽取到 {len(table_tpl)} 张表结构模板、{len(metrics_tpl)} 个指标")

    records: list[dict] = []

    # ---- 1) 问数 → SQL：统一 system ----
    for r in src_sql + src_pipe:
        conv = r.get("conversations")
        if not conv or len(conv) < 3:
            continue
        records.append({
            "task": "generate_sql",
            "question": r.get("question"),
            "source": r.get("source", "pipeline"),
            "conversations": [
                {"from": "system", "value": UNIFIED_SYSTEM},
                {"from": "human", "value": conv[1]["value"]},
                {"from": "gpt", "value": conv[2]["value"]},
            ],
        })

    # ---- 2) 纠错任务：保持它自己的 system（SQL 纠错专家）----
    for r in src_correct:
        conv = r.get("conversations")
        if not conv or len(conv) < 3:
            continue
        records.append({
            "task": "correct_sql",
            "question": r.get("question"),
            "source": "correct_sql",
            # 保留错误类型，供下面的配比阶段按类型均匀抽样
            "error_type": r.get("error_type"),
            "conversations": conv,
        })

    # ---- 3) 澄清/拒答：统一 system + **补齐缺失的 schema 上下文** ----
    fixed_neg = 0
    for r in src_neg:
        conv = r.get("conversations")
        if not conv or len(conv) < 3:
            continue
        q = r.get("question") or ""
        old_human = conv[1]["value"]
        if "【可用数据表信息】" not in old_human:
            human = build_human(q, table_tpl, metrics_tpl)
            fixed_neg += 1
        else:
            human = old_human
        records.append({
            "task": "clarify" if not r.get("refuse") else "refuse",
            "question": q,
            "source": r.get("source"),
            "conversations": [
                {"from": "system", "value": UNIFIED_SYSTEM},
                {"from": "human", "value": human},
                {"from": "gpt", "value": conv[2]["value"]},
            ],
        })
    print(f"[2/4] 合并 {len(records)} 条；其中为 {fixed_neg} 条负样本补齐了 schema 上下文")

    # ---- 去重 ----
    # 注意去重键的选取：
    #   * 问数/澄清/拒答：同任务 + 同问题 视为重复（答案唯一）
    #   * 纠错：同一个问题会对应**多种不同的错误 SQL**（字段名错/表名错/语法错…），
    #     每种都是独立有效样本，因此须按「错误SQL」而不是「问题」去重。
    #     早期版本按问题去重，把 541 条纠错样本误并成 121 条。
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in records:
        if r["task"] == "correct_sql":
            key = ("correct_sql", (r["conversations"][1]["value"]).strip())
        else:
            key = (r["task"], (r.get("question") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"      去重后 {len(deduped)} 条（移除 {len(records) - len(deduped)} 条重复）")

    # ---- 类别配比 ----
    # 生成一条 SQL 只调一次 generate_sql，纠错是概率触发的（EXPLAIN 失败才走）。
    # 若让纠错样本占一半，模型会偏向"改错"而不是"写对"，因此对 correct_sql 降采样。
    # 按错误类型均匀抽取，避免只留下某一种错误。
    gen = [r for r in deduped if r["task"] == "generate_sql"]
    cor = [r for r in deduped if r["task"] == "correct_sql"]
    oth = [r for r in deduped if r["task"] not in ("generate_sql", "correct_sql")]

    # 澄清样本也做配比：它样本量大但属于少数派任务，
    # 占比过高会挤占 SQL 生成能力，过低则学不会反问。
    # 目标：clarify 占训练集约 clarify-ratio（默认 15%）。
    # 按缺失组合分层抽样，保证各组合都有覆盖。
    cl = [r for r in oth if r["task"] == "clarify"]
    rf = [r for r in oth if r["task"] != "clarify"]
    if args.clarify_ratio > 0 and gen and cl:
        target_cl = int(len(gen) * args.clarify_ratio)
        if target_cl < len(cl):
            by_missing: dict[tuple, list[dict]] = {}
            for r in cl:
                # 缺失组合从 human 段的编号行体现，这里用答案文本 + 问题做粗略分层：
                # 直接用 missing 字段更准 —— build 阶段没保留它，故从答案推断
                ans = r["conversations"][2]["value"]
                key = ans.split("\n", 1)[1] if "\n" in ans else ans
                by_missing.setdefault(key, []).append(r)
            rng_c = random.Random(args.seed)
            picked: list[dict] = []
            per = max(1, target_cl // max(1, len(by_missing)))
            for k, rows in by_missing.items():
                rng_c.shuffle(rows)
                picked.extend(rows[:per])
            if len(picked) < target_cl:
                rest = [r for r in cl if r not in picked]
                rng_c.shuffle(rest)
                picked.extend(rest[: target_cl - len(picked)])
            print(f"      clarify 降采样：{len(cl)} -> {len(picked)} 条"
                  f"（覆盖 {len(by_missing)} 种缺失组合，比例 {args.clarify_ratio:g}:1）")
            cl = picked
    oth = cl + rf
    if args.correct_ratio > 0 and cor:
        target = int(len(gen) * args.correct_ratio)
        if target < len(cor):
            by_type: dict[str, list[dict]] = {}
            for r in cor:
                # 错误类型存在源数据的 error_type 字段里（形如"字段名拼写错误: a -> b"），
                # 取冒号前的类别名作为分组键
                et = r.get("error_type") or "其他"
                key = et.split(":")[0].strip()[:40]
                by_type.setdefault(key, []).append(r)
            rng2 = random.Random(args.seed)
            picked: list[dict] = []
            # 每类先按比例取，保证类型均衡
            per = max(1, target // max(1, len(by_type)))
            for k, rows in by_type.items():
                rng2.shuffle(rows)
                picked.extend(rows[:per])
            # 不足则从剩余里补
            if len(picked) < target:
                rest = [r for r in cor if r not in picked]
                rng2.shuffle(rest)
                picked.extend(rest[: target - len(picked)])
            print(f"      correct_sql 降采样：{len(cor)} -> {len(picked)} 条"
                  f"（覆盖 {len(by_type)} 种错误类型，比例 {args.correct_ratio:g}:1）")
            cor = picked
    deduped = gen + cor + oth
    rng3 = random.Random(args.seed)
    rng3.shuffle(deduped)

    # ---- 划分 train / val ----
    rng = random.Random(args.seed)
    rng.shuffle(deduped)
    n_val = max(1, int(len(deduped) * args.val_ratio))
    val = deduped[:n_val]
    train = deduped[n_val:]

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_dir: Path = args.out_dir
    train_path = out_dir / "train_final.jsonl"
    val_path = out_dir / "val_final.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    # ---- LLaMA-Factory dataset_info.json ----
    dataset_info = {
        "zhangguiwenshu_sft": {
            "file_name": "train_final.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system",
            },
        },
        "zhangguiwenshu_val": {
            "file_name": "val_final.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system",
            },
        },
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 统计 ----
    task_counter = Counter(r["task"] for r in deduped)
    print(f"[3/4] 写出：{train_path.name} {len(train)} 条 / {val_path.name} {len(val)} 条")
    print(f"      dataset_info.json（LLaMA-Factory 注册用）")
    print()
    print("[4/4] 任务类型分布：")
    for k, v in task_counter.most_common():
        print(f"      {k:<14} {v:>4} 条  ({v/len(deduped)*100:.1f}%)")
    print()
    print(f"      合计 {len(deduped)} 条")
    print()
    print("提醒：正确回答（SQL）类占绝大多数，澄清/拒答类是少数 —— 这个比例是合理的，")
    print("      但要确认模型没有因为样本少而忽略澄清行为（训练后看 L3 指标）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
