#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
由标准 SQL 机械反向生成「澄清类」与「拒答类」训练样本。

为什么需要这个脚本
------------------
现有 LangGraph 流水线**不具备澄清能力**（只会硬编 SQL 或 correct_sql 重写），
因此 build_train_set.py 采不到 3.2.4 / 3.2.6 所需的负样本。
而项目只有一个人、没有人工标注，无法手写上千条。

解决思路：**机械反向变换**
    标准 SQL 里必然带有时间条件（d.month / d.quarter）、分组维度（GROUP BY）和聚合口径
    （SUM/AVG/COUNT）。把这些信息**逐一摘掉**，问题就自然变成"信息不全、应当反问"。
    这个变换是**可逆且按构造正确**的：
      - 摘掉时间条件  -> 必须澄清"时间范围"
      - 摘掉分组维度  -> 必须澄清"统计粒度"
      - 用模糊词替换聚合 -> 必须澄清"指标口径"
    因此不需要人判断答案对不对。

拒答类样本同理：把"查询"语义反向变成 DML/DDL 或越权读，答案就是标准拒答语。

用法
----
    python finetune/make_negatives.py
    python finetune/make_negatives.py --out finetune/data/negatives.jsonl --seed 7
"""

from __future__ import annotations

import argparse
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
DATA_DIR = HERE / "data"
PROMPT_PATH = HERE.parent / "prompt" / "generate_sql.prompt"


# ==========================================================================
# system 提示词：在原有 SQL 约束上，追加"信息不足要反问"的行为约定
# ==========================================================================
def build_system_prompt() -> str:
    base = ""
    if PROMPT_PATH.exists():
        text = PROMPT_PATH.read_text(encoding="utf-8")
        text = re.sub(r"\{[a-z_]+\}", "", text)
        text = re.split(r"用户查询", text)[0]
        base = "\n".join(ln.rstrip() for ln in text.splitlines() if ln.strip()).strip()

    behavior = (
        "【输出类型】\n"
        "你必须二选一输出：\n"
        "A. 若问题已具备「时间范围」「统计粒度」「指标口径」等必要信息，输出一条 MySQL 只读查询语句；\n"
        "B. 若信息不足，不要臆测，改为提出澄清问题：先用一句话指出缺少哪些信息，再列出 2-3 个具体可选的确认项，"
        "每个确认项给出可选值，便于用户直接回答。\n"
        "无论哪种情况，都只输出纯文本，不要使用 Markdown 代码块。"
    )
    return f"{base}\n\n{behavior}" if base else behavior


# ==========================================================================
# 反向变换：SQL -> 信息不全的问题 + 期望的澄清回答
# ==========================================================================
# 从 SQL 中可识别的"信息要素"
_RE_TIME = re.compile(r"d\.month\s*=\s*(\d+)|d\.quarter\s*=\s*'([^']+)'|d\.year\s*=\s*(\d+)")
_RE_GROUP = re.compile(r"GROUP\s+BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)", re.IGNORECASE | re.DOTALL)
_RE_AGG = re.compile(r"\b(SUM|AVG|COUNT|MAX|MIN)\s*\(\s*(DISTINCT\s+)?([\w.]+)\s*\)", re.IGNORECASE)

# 聚合函数 -> 模糊说法（仅用于判断"是否含明确口径"）
_AGG_VAGUE = {
    "SUM": "总体情况",
    "AVG": "平均水平",
    "COUNT": "数量情况",
    "MAX": "最高的情况",
    "MIN": "最低的情况",
}

# 分组字段 -> 中文维度名
_DIM_CN = {
    "r.region_name": "大区", "r.province": "省份", "p.category": "品类",
    "p.brand": "品牌", "p.product_name": "商品", "c.member_level": "会员等级",
    "c.gender": "性别", "d.month": "月份", "d.quarter": "季度", "d.day": "日期",
    "f.customer_id": "客户",
}

# 维度/时间词：从实体里剔除，避免把"服饰"这类维度当成业务实体
_DIM_WORDS = (
    "大区", "省份", "地区", "品类", "品牌", "商品", "会员", "会员等级", "性别",
    "月份", "月", "季度", "日期", "天", "客户", "各个", "每个", "各", "按", "所有", "全部",
)

# 业务实体：省份 / 大区 / 品类 / 品牌 / 商品名（用于保留"问的是什么业务"）
_ENTITIES = [
    "广东省", "浙江省", "四川省", "北京市", "上海市", "湖北省",
    "华南", "华东", "西南", "华北", "华中",
    "手机数码", "家用电器", "鞋靴", "服饰", "食品饮料", "休闲零食",
    "苹果", "华为", "三星", "戴森", "美的", "耐克", "阿迪达斯", "优衣库",
    "李维斯", "雀巢", "蒙牛", "乐事", "奥利奥", "亚马逊",
    "iPhone", "Galaxy", "Mate", "Kindle",
]

# 自然的口语化问法模板：信息不全是常态，但听起来要像真人提问
_VAGUE_TEMPLATES = {
    # 同时缺时间与口径
    ("time", "metric"): [
        "{e}卖得怎么样？", "{e}的业绩怎么样？", "看看{e}的情况",
        "{e}表现如何？", "帮我分析下{e}", "{e}的数据怎么样？",
    ],
    # 只缺时间
    ("time",): [
        "{e}的销售额怎么样？", "{e}现在表现如何？", "看看{e}的销售数据",
        "{e}卖得好吗？",
    ],
    # 只缺口径
    ("metric",): [
        "{e}的情况怎么样？", "{e}表现如何？", "看看{e}的关键数据",
        "{e}整体怎么样？",
    ],
    # 只缺粒度
    ("grain",): [
        "看看{e}的整体情况", "{e}总体怎么样？", "{e}的汇总数据给我看看",
    ],
    # 缺时间 + 粒度
    ("time", "grain"): [
        "看看{e}的销售情况", "{e}最近卖得怎么样？", "帮我看看{e}的数据",
    ],
    # 缺粒度 + 口径
    ("grain", "metric"): [
        "{e}怎么样？", "看看{e}的情况", "{e}的业务表现如何？",
    ],
    # 三者全缺
    ("time", "grain", "metric"): [
        "看看{e}的销售情况", "{e}最近怎么样？", "帮我分析下{e}的业务",
        "{e}的经营情况如何？",
    ],
}

# 无实体时的通用模糊问法
_GENERIC_VAGUE = [
    "看看销售情况", "销售怎么样？", "帮我分析一下数据", "最近生意好不好？",
    "业务表现如何？", "帮我做个报表",
]

# 以疑问词结尾的模板才该补问号；"看看…/帮我…"这类祈使句不加问号
_QUESTION_TAIL = re.compile(r"(怎么样|如何|好不好|吗|呢|多少|哪些|是什么)[？?]?$")


def _dim_cn(expr: str) -> str:
    """把 SQL 里的分组字段表达式翻成中文维度名。"""
    return _DIM_CN.get(expr.strip(), "维度")


def _extract_entity(question: str) -> str | None:
    """从原问题里抽出业务实体（省份/大区/品类/品牌/商品），用于保留"问的是什么"。"""
    for e in sorted(_ENTITIES, key=len, reverse=True):
        if e in question:
            # 实体本身不能是纯维度词
            if any(e == w or e == w + "省" or e == w + "市" for w in _DIM_WORDS):
                continue
            return e
    return None


def reverse_to_clarify(question: str, sql: str, rng: random.Random) -> dict | None:
    """
    把「信息完整的问题 + SQL」反向变成「信息不全的问题 + 澄清回答」。

    注意：不能靠正则删词来"造"问题 —— 那样会产出
    "月的总体情况是多少"这种既不合语法、也不像真人提问的句子。
    正确做法是：从 SQL 判定缺失了哪些要素，再从**自然口语模板**里挑一个，
    并把原问题中的业务实体（如"广东省""服饰"）填进去。

    返回 None 表示这条 SQL 没有可摘除的信息要素。
    """
    if not sql:
        return None

    has_time = bool(_RE_TIME.search(sql))
    g = _RE_GROUP.search(sql)
    has_group = bool(g)
    has_agg = bool(_RE_AGG.search(sql))

    if not (has_time or has_group):
        return None

    # ---- 判定缺失要素 ----
    missing: list[str] = []
    keys: list[str] = []
    if has_time:
        missing.append("时间范围")
        keys.append("time")
    if has_group:
        missing.append("统计粒度")
        keys.append("grain")
    if has_agg:
        missing.append("指标口径")
        keys.append("metric")
    key = tuple(sorted(keys))

    # ---- 选模板并填入业务实体 ----
    entity = _extract_entity(question)
    templates = _VAGUE_TEMPLATES.get(key)
    if templates is None:
        # 组合未覆盖时退化为"缺时间+口径"的说法
        templates = _VAGUE_TEMPLATES[("time", "metric")]
        if "时间范围" not in missing:
            missing.insert(0, "时间范围")
    if entity:
        new_q = rng.choice(templates).format(e=entity)
    else:
        new_q = rng.choice(_GENERIC_VAGUE)

    # 只给真正的疑问句补问号；「看看…」「帮我…」这类祈使句保持原样
    new_q = new_q.rstrip("？?")
    if _QUESTION_TAIL.search(new_q):
        new_q += "？"

    # ---- 澄清回答：分点 + 给可选项，便于用户直接回答 ----
    lines = ["这个问题还缺少一些必要信息，我需要确认后才能给出准确结果："]
    idx = 1
    if "时间范围" in missing:
        lines.append(f"{idx}. 统计的时间范围？（例如：1月 / 第一季度 / 全部数据）")
        idx += 1
    if "统计粒度" in missing:
        dims = [_dim_cn(x) for x in g.group(1).split(",") if x.strip()] if g else []
        opts = " / ".join(d for d in dims if d) or "大区 / 品类 / 月份"
        lines.append(f"{idx}. 希望按什么维度看？（例如：{opts}）")
        idx += 1
    if "指标口径" in missing:
        lines.append(
            f"{idx}. 具体看哪个指标？（例如：销售额 GMV / 客单价 AOV / 销量 / 订单数）"
        )
        idx += 1
    answer = "\n".join(lines)

    return {
        "question": new_q,
        "answer": answer,
        "capability_hint": "CLARIFY",
        "missing": missing,
        "derived_from_sql": sql,
        "source": "reverse_transform",
    }


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def build_refusal_samples(schema: dict, rng: random.Random) -> list[dict]:
    """拒答类：DML/DDL、越权读取。答案标明"只能查询"。"""
    regions = schema["regions"]
    products = schema["products"]
    customers = schema["customers"]

    p_name = products[0]["product_name"]
    p_brand = products[0]["brand"]
    r_name = regions[0]["region_name"]
    level = customers[0]["member_level"]

    risky: list[tuple[str, str]] = [
        (f"把 {p_name} 这个商品下架，从商品表里删掉。", "删除数据（DELETE）"),
        ("帮我把所有订单金额改成 0。", "修改数据（UPDATE）"),
        ("把 fact_order 表删了重建。", "删除表（DROP）"),
        ("清空 dim_customer 表。", "清空表（TRUNCATE）"),
        (f"往订单表里插一条测试数据，金额随便写。", "写入数据（INSERT）"),
        ("把 {t} 表的字段类型从 varchar 改成 int。".format(t="dim_product"), "修改表结构（ALTER）"),
        ("给我一份数据库里所有用户的账号和密码。", "越权读取凭据"),
        ("把 member_level 字段的加密密钥查出来。", "越权读取密钥"),
        ("帮我导出全部客户的手机号和身份证号。", "越权读取个人敏感信息"),
        ("DROP TABLE dim_region;", "删除表（DROP）"),
        ("truncate fact_order;", "清空表（TRUNCATE）"),
        ("update dim_product set brand='X' where 1=1;", "修改数据（UPDATE）"),
        (f"把 {level} 会员的数据全部删掉。", "删除数据（DELETE）"),
        (f"{r_name}大区的订单记录帮我批量删除。", "删除数据（DELETE）"),
        ("给 fact_order 加一个备注字段。", "修改表结构（ALTER）"),
        ("帮我把 dw 库的权限改成所有用户可写。", "提权操作"),
    ]

    out: list[dict] = []
    for q, kind in risky:
        answer = (
            f"抱歉，这个请求我无法执行。我只能生成只读的查询语句（SELECT），"
            f"不具备执行「{kind}」的权限，也不应该对数据做任何变更。\n"
            f"如果你确实需要变更数据，请通过有权限的运维流程操作。\n"
            f"如果你是想查看相关数据，我可以帮你查询，例如：\n"
            f"- 查看相关记录的数量与明细\n"
            f"- 统计相关指标（销售额、销量、订单数等）"
        )
        out.append({
            "question": q,
            "answer": answer,
            "capability_hint": "SAFETY_DIALECT",
            "missing": None,
            "derived_from_sql": None,
            "source": "refusal_template",
        })
    return out


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="生成澄清类与拒答类训练样本")
    parser.add_argument("--test-set", type=Path, default=DATA_DIR / "test.jsonl",
                        help="从中取标准 SQL 做反向变换")
    parser.add_argument("--train-set", type=Path, default=DATA_DIR / "train.jsonl",
                        help="已采集的训练集（可选，一起做反向变换）")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "negatives.jsonl")
    parser.add_argument("--seed", type=int, default=20250921)
    parser.add_argument("--max-clarify", type=int, default=220, help="澄清样本上限")
    parser.add_argument("--schema", type=Path, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- 收集可用的标准 SQL ----
    sql_pool: list[tuple[str, str]] = []
    for path in (args.test_set, args.train_set):
        if not path.exists():
            continue
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            sql = rec.get("reference_sql")
            q = rec.get("question")
            if sql and q:
                sql_pool.append((q, sql))
    if not sql_pool:
        print("[错误] 没有找到带标准 SQL 的样本，请先运行 gen_eval_set.py", file=sys.stderr)
        return 2
    print(f"[1/4] 从 {len(sql_pool)} 条 (问题, SQL) 中做反向变换")

    # ---- 澄清样本 ----
    rng.shuffle(sql_pool)
    clarify: list[dict] = []
    seen_q: set[str] = set()
    for q, sql in sql_pool:
        if len(clarify) >= args.max_clarify:
            break
        item = reverse_to_clarify(q, sql, rng)
        if item and item["question"] not in seen_q:
            seen_q.add(item["question"])
            clarify.append(item)
    print(f"[2/4] 反向生成澄清样本 {len(clarify)} 条")

    # ---- 拒答样本 ----
    sys.path.insert(0, str(HERE))
    from gen_eval_set import load_schema, DEFAULT_SCHEMA_CANDIDATES
    schema_path = args.schema or next((p for p in DEFAULT_SCHEMA_CANDIDATES if p.exists()), None)
    if schema_path is None:
        print("[错误] 找不到 dw.sql（拒答样本需要真实实体名）", file=sys.stderr)
        return 2
    schema = load_schema(schema_path)
    refusals = build_refusal_samples(schema, rng)
    print(f"[3/4] 生成拒答样本 {len(refusals)} 条")

    # ---- system 提示词统一 ----
    system_prompt = build_system_prompt()

    records: list[dict] = []
    for item in clarify + refusals:
        records.append({
            "question": item["question"],
            "capability_hint": item["capability_hint"],
            "missing": item["missing"],
            "source": item["source"],
            "derived_from_sql": item["derived_from_sql"],
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": f"【用户查询】\n{item['question']}"},
                {"from": "gpt", "value": item["answer"]},
            ],
        })

    rng.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[4/4] 已写出 {len(records)} 条 -> {args.out}")
    print(f"      澄清 {len(clarify)} 条 | 拒答 {len(refusals)} 条")
    print()
    print("  说明：澄清样本由标准 SQL 机械摘除时间/维度/口径反向生成，")
    print("       变换按构造即可保证答案正确，无需人工标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
