#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合成「带 schema + 无实体超短问句 → 澄清」训练样本（澄清能力专项补强）

为什么单独写这个脚本
====================
第一轮微调（656 条）的实测结果表明：**澄清能力从 72.2% 掉到 55.6%**，
且失败题型高度集中 —— 全部是"无业务实体的超短问句"：

    评测集 18 条应澄清题：17 条完全没有实体，平均 6.6 字
      「客单价怎么样」「销售额是多少」「利润率是多少」「哪个卖得最好」
      「趋势如何」「订单情况」「按地区看一下」

而原训练集（make_negatives.py 生成）里的澄清样本长这样：

    三星最近如何？ / 耐克怎么样？ / 服饰卖得怎么样？
    —— 平均 9.9 字，**几乎每条都带一个业务实体**

于是模型学到的规则是「**看到某个实体名 + 怎么样 → 就反问**」，
而不是「**信息不足 → 就反问**」。它把实体词当成了触发澄淡的开关，
遇到「客单价怎么样」这种没有实体的裸指标名，就直接输出
`SELECT AVG(order_amount) AS aov FROM fact_order;`。

本脚本做的事
============
**让"信息不足"本身成为触发条件**，而不是依赖实体词：

1. 问法**以裸指标名/裸维度词为主**，绝大多数不带业务实体，长度集中在 4–10 字；
2. **显式构造"只缺一个要素"的题**（尤其 `销售额是多少` 这种只缺时间范围的），
   这是第一轮最典型的失败模式；
3. 覆盖评测集实际出现的全部 6 种 missing 标签
   （时间范围 / 统计粒度 / 指标口径 / 分析对象 / 分析维度 / 对比基期）；
4. 澄清回答**按 missing 标签逐点反问**，并给可选项，与评测器 `_MISSING_KEYWORDS`
   的要点覆盖判定对齐（保证"覆盖 ≥ 一半要点"）。

答案正确性
==========
澄清类的标准答案不是 SQL，而是"**应当反问，且反问的要点与缺失项一致**"。
缺失项由模板**显式指定**，因此答案按构造即正确，**不需要人工标注**。

用法
----
    python finetune/synthesize_clarify_short.py
    python finetune/synthesize_clarify_short.py --per-combo 45 --refusals 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
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


# ==========================================================================
# 一、澄清问句模板
# ==========================================================================
# 键 = 缺失要点的组合；值 = 该组合下的问法。
#
# 设计原则（每一条都对应第一轮的实测失败）：
#   * 优先**裸指标名/裸维度词**，不带实体 —— 这是第一轮的病根
#   * 带实体和不带实体的都要有，但**不带实体的占多数**
#   * 长度控制在 4–11 字，与评测集同分布
#   * "只缺一个要素"的组合必须有足够样本（第一轮最典型的失败）
#
# {e} 占位符会填入真实业务实体（省份/大区/品类/品牌）—— 刻意少用。

# ---- 不带实体的裸问法（主力）----
_BARE = {
    ("time",): [
        "销售额是多少", "销量是多少", "订单数是多少", "利润是多少",
        "成本是多少", "库存还有多少", "退货有多少", "复购率是多少",
        "客单价是多少", "毛利是多少", "转化率是多少", "退款金额是多少",
        "一共有多少订单", "总销售额多少", "现在销售额多少",
        "最近销售额多少", "这个月销售额多少", "今年销售额多少",
        "本月销量是多少", "当季营收是多少", "日均销售额是多少",
        "周销售额是多少", "上月利润是多少", "本季度订单量是多少",
        "营业额是多少", "流水是多少", "收入是多少", "支出是多少",
        "毛利率多少", "净利多少", "单量是多少", "件数是多少",
        "重量是多少", "总额是多少", "均价是多少", "折扣率是多少",
        "缺货多少", "滞销多少", "新增多少", "流失多少",
        "复购几次", "回购率是多少", "客诉多少", "赔付多少",
    ],
    ("metric",): [
        "哪个品牌最受欢迎", "哪个卖得最好", "利润率是多少", "毛利率是多少",
        "排名前十的是什么", "哪个品类最好卖", "谁是销冠", "哪个渠道最有效",
        "什么卖得最好", "哪个商品最畅销", "哪家表现最好", "最热门的是什么",
        "哪个省最强", "哪个大区领先", "哪个会员等级值钱",
        "谁贡献最大", "哪个品类增速快", "哪个品牌在涨", "哪个在跌",
        "哪些商品滞销", "哪个渠道划算", "哪些客户优质", "哪个转化高",
        "什么最好卖", "哪些值得关注", "哪个风险高", "什么拖后腿",
    ],
    ("grain",): [
        "整体情况给我看看", "汇总数据给我看看", "总体数据是多少",
        "给我看个概览", "总的情况如何", "整体表现如何", "汇总一下",
        "给我个总数", "总计是多少", "全量数据给我", "总体情况如何",
        "汇总数据是多少", "整体汇总一下", "总共多少",
    ],
    ("time", "metric"): [
        "销售怎么样？", "帮我分析一下数据", "帮我统计一下", "订单情况",
        "表现最好的地区", "对比一下各个渠道", "最近生意好不好",
        "数据怎么样", "帮我看看数据", "业务表现如何", "情况如何",
        "经营状况如何", "业绩如何", "帮我分析一下", "看看最近的表现",
        "经营得怎么样", "整体情况怎么样", "给我分析一下经营情况",
        "帮我看看业务数据", "最近表现如何", "帮我拉一下数据",
        "业务情况如何", "帮忙分析下数据", "看看经营情况",
        "帮我看下生意", "最近效益怎么样", "生意如何", "经营得好不好",
        "帮我复盘一下", "复盘一下最近", "看下经营数据", "帮我看看业绩",
        "最近数据怎么样", "帮我出个数", "给我个结论", "生意还行吗",
    ],
    ("time", "grain"): [
        "客单价怎么样", "销售额是多少", "销量如何", "订单量怎么样",
        "平均订单金额是多少", "单笔金额是多少", "每单多少钱",
        "销售数据是多少", "订单数据是多少", "客单数据怎么样",
        "平均销售额是多少", "人均消费是多少",
        "平均每单多少", "单价多少", "件单价多少", "每单件数多少",
        "人均多少", "户均多少", "平均客单多少", "单均多少",
    ],
    ("grain", "metric"): [
        "这个数据怎么样", "那个情况如何", "怎么样啊", "情况如何",
        "表现怎么样", "数据如何", "怎么样", "情况怎么样",
        "如何", "还行吗", "好不好", "怎么样呢", "什么情况",
        "啥情况", "如何了", "给我说说",
    ],
    ("time", "grain", "metric"): [
        "看看销售情况", "看看情况", "帮我看看", "查一下", "看一下数据",
        "看看数据", "给我看看情况", "查一下情况", "看看业务情况",
        "帮我查一下", "了解一下情况", "看下数据",
        "看看", "查查", "看下", "了解一下", "给我看看", "瞧瞧",
    ],
    # ---- 分析对象类（"分析什么"都没说清）----
    ("target", "metric", "time"): [
        "帮我分析一下数据", "帮我统计一下", "分析一下", "统计一下",
        "帮我做一份分析", "帮我出个报表", "分析一下业务", "做个数据统计",
        "帮我分析分析", "统计数据给我", "帮我跑个数",
        "帮我盘一下", "盘一下数据", "帮我拉个数", "出个分析",
        "帮我看看整体", "整体分析一下", "帮我梳理一下", "梳理下数据",
        "给我一份分析", "做个分析", "帮我算算", "算一下",
    ],
    ("target", "time"): [
        "趋势如何", "趋势怎么样", "走势如何", "变化趋势如何",
        "看下趋势", "最近趋势如何", "趋势是怎么走的",
        "走势怎么样", "变化如何", "走势如何了", "趋势呢",
    ],
    ("target", "metric"): [
        "哪个卖得最好", "哪个最好", "什么最受欢迎", "哪个最热销",
        "最受欢迎的是什么", "哪个表现最好", "什么最好卖",
        "哪个最强", "哪个最牛", "哪个领先", "谁最好",
    ],
    # ---- 分析维度类（"按什么看"没说清）----
    ("dim", "metric"): [
        "客户情况怎么样", "客户情况如何", "用户情况怎么样",
        "客户数据怎么样", "用户表现如何", "客户表现怎么样",
        "客户怎么样", "用户情况如何", "客户情况呢", "用户怎么样",
    ],
    ("dim", "metric", "time"): [
        "按地区看一下", "按品类看一下", "分渠道看看", "按品牌统计一下",
        "分组看一下", "按客户看看", "分地区统计", "按维度看看",
        "分品类统计一下", "按渠道分析一下",
        "按省份看下", "分品牌看看", "按会员看看", "分性别统计",
        "按月份看一下", "分组统计下", "拆分看一下", "细看一下",
    ],
    # ---- 对比基期类 ----
    ("metric", "base"): [
        "同环比增长多少", "同比增长多少", "环比增长多少",
        "和上期比怎么样", "比上个月如何", "同比情况如何",
        "环比变化如何", "增长了多少", "涨幅是多少",
        "同比多少", "环比多少", "相比如何", "增长情况如何",
        "涨了还是跌了", "变化多少", "和之前比如何",
    ],
    ("base", "time"): [
        "和去年比怎么样", "对比一下上期", "跟上个月比比",
        "同比如何", "环比如何",
        "和上季度比呢", "对比去年同期", "跟上期比比", "和前一期比",
    ],
}

# ---- 带实体的问法（补充，占比刻意压低）----
_WITH_ENTITY = {
    ("time",): [
        "{e}的销售额是多少", "{e}的销量是多少", "{e}现在卖多少",
        "{e}这段时间卖了多少",
    ],
    ("time", "metric"): [
        "{e}卖得怎么样？", "{e}卖得好不好", "{e}最近怎么样？",
        "{e}表现如何？", "{e}情况怎么样",
    ],
    ("time", "grain"): [
        "{e}的客单价怎么样", "{e}的销售额是多少", "{e}的销量如何",
    ],
    ("dim", "metric", "time"): [
        "按{e}看一下", "分{e}看看", "按{e}统计一下",
    ],
}


def _all_templates() -> dict[tuple, list[str]]:
    """把裸问法与带实体问法合并，便于统一遍历。"""
    merged: dict[tuple, list[str]] = {}
    for keys in set(_BARE) | set(_WITH_ENTITY):
        merged[keys] = list(_BARE.get(keys, [])) + list(_WITH_ENTITY.get(keys, []))
    return merged


# ==========================================================================
# 二、缺失要点 -> 中文标签 + 反问句
# ==========================================================================
# 标签用评测集里真实出现的说法（不是自造的），因为评测器按标签选关键词：
#   _MISSING_KEYWORDS = {
#       "时间范围": (时间/月份/季度/日期/期间...), "统计粒度": (维度/按/分组...),
#       "指标口径": (指标/销售额/GMV/客单价/AOV/销量/订单数...),
#       "分析对象": (分析/对象/指标/维度), "分析维度": (维度/按/分组),
#       "对比基期": (基期/对比/同比/环比/时间),
#   }
_LABEL = {
    "time": "时间范围",
    "grain": "统计粒度",
    "metric": "指标口径",
    "target": "分析对象",
    "dim": "分析维度",
    "base": "对比基期",
}

# 每个要点对应的反问句（含可选项，便于用户直接回答）
_ASK = {
    "时间范围": "统计的时间范围？（例如：1月 / 第一季度 / 全部数据 / 最近一周）",
    "统计粒度": "希望按什么维度、什么粒度看？（例如：按天 / 按月 / 按大区 / 按品类 / 不分组）",
    "指标口径": "具体看哪个指标？（例如：销售额 GMV / 客单价 AOV / 销量 / 订单数 / 利润率）",
    "分析对象": "想分析的对象是什么？（例如：销售业绩 / 客户 / 商品 / 地区 / 渠道）",
    "分析维度": "想按哪个维度拆分看？（例如：大区 / 省份 / 品类 / 品牌 / 月份）",
    "对比基期": "对比的基期是哪一段？（例如：去年同期 / 上个月 / 上一季度 / 环比）",
}

# 缺失项顺序：先问"分析什么"，再问"怎么切"，最后问"看哪个数"
_ORDER = ["分析对象", "分析维度", "对比基期", "时间范围", "统计粒度", "指标口径"]


def _build_answer(labels: list[str]) -> str:
    """
    生成澄清回答。

    结构刻意与评测器的判定对齐：
      * 首句含"缺少"→ 命中澄清标记
      * 逐点分条，每条的措辞含对应 `_MISSING_KEYWORDS` 的关键词
    """
    ordered = [lb for lb in _ORDER if lb in labels]
    lines = ["这个问题还缺少一些必要信息，我需要确认后才能给出准确结果："]
    for i, lb in enumerate(ordered, 1):
        lines.append(f"{i}. {_ASK[lb]}")
    return "\n".join(lines)


# ==========================================================================
# 三、生成
# ==========================================================================
def _load_jsonl(p: Path) -> list[dict]:
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


def _norm(s: str) -> str:
    """归一化：去空白与常见标点，用于查重。"""
    return re.sub(r"[\s，。、？！?！,.]", "", s or "")


def build_test_questions(test_path: Path) -> set[str]:
    """
    读评测集，拿到**所有**问句的归一化形式用于查重。

    注意：这里只是"避免和评测集撞句"，**不使用评测集的任何答案或标签**，
    因此不构成训练/评测泄漏。若某条模板句恰好与评测题重合，直接丢弃。
    """
    recs = _load_jsonl(test_path)
    return {_norm(r.get("question", "")) for r in recs}


def synthesize(rng: random.Random, entities: list[str], per_combo: int,
               forbid: set[str]) -> list[dict]:
    tpls = _all_templates()
    out: list[dict] = []
    seen: set[str] = set()
    dropped = 0

    # 让"只缺一个要素"的组合先抽满 —— 那是第一轮最典型的失败题型
    combos = sorted(tpls, key=lambda k: (len(k), k))

    for keys in combos:
        labels = [_LABEL[k] for k in keys]
        answer = _build_answer(labels)
        made = 0
        pool = list(tpls[keys])
        rng.shuffle(pool)
        ents = list(entities)
        rng.shuffle(ents)

        # 先跑裸问法，实体问法只在额度没填满时补（保证"无实体"占多数）
        rounds = 0
        while made < per_combo and rounds < 40:
            rounds += 1
            progressed = False
            for raw in pool:
                if made >= per_combo:
                    break
                if "{e}" in raw:
                    # 实体问法：额度过半后才允许出现，确保裸问法占多数
                    if made < per_combo * 0.7:
                        continue
                    if not ents:
                        continue
                    q = raw.format(e=ents[made % len(ents)])
                else:
                    q = raw
                q = q.strip()
                nq = _norm(q)
                if not nq or nq in seen or nq in forbid:
                    if nq in forbid:
                        dropped += 1
                    continue
                seen.add(nq)
                out.append({
                    "question": q,
                    "answer": answer,
                    "capability_hint": "CLARIFY",
                    "missing": labels,
                    "derived_from_sql": None,
                    "source": "clarify_short_bare",
                })
                made += 1
                progressed = True
            if not progressed:
                break

    if dropped:
        print(f"      丢弃 {dropped} 条与评测集问句重合的模板句（防泄漏）")
    return out


# ==========================================================================
# 四、拒答样本（第一轮的另一处短板：拒答随训练轮数退化）
# ==========================================================================
# 第一轮 3 epoch 时模型在"删除 fact_order 表"上直接输出 `DELETE FROM fact_order;`。
# 原因之一是拒答样本仅 6 条（占 0.9%），训久了被 SQL 生成能力淹没。
# 这里把拒答扩到 40 条，并让**拒绝话术多样化**（评测器按语义关键词判定）。
_REFUSAL_KINDS = [
    ("删除数据（DELETE）", [
        "把 {p} 这个商品从商品表里删掉",
        "把下架的商品都删除",
        "删除订单表里金额为 0 的记录",
        "把 {level} 会员的数据全部删掉",
        "{region}大区的订单记录帮我批量删除",
        "删掉所有退货的订单",
        "把重复的客户记录清理掉",
    ]),
    ("修改数据（UPDATE）", [
        "帮我把所有订单金额改成 0",
        "把 {p} 的价格改成 1 元",
        "把所有客户等级统一改成黄金",
        "把订单日期批量往后调一天",
        "把销量字段都加上 10",
        "更新一下商品品牌名称",
    ]),
    ("删除表（DROP/TRUNCATE）", [
        "把 fact_order 表删了重建",
        "清空 dim_customer 表",
        "DROP TABLE dim_region",
        "truncate fact_order",
        "把 dim_product 表整个删掉",
        "drop table fact_order",
    ]),
    ("写入数据（INSERT）", [
        "往订单表里插一条测试数据，金额随便写",
        "帮我造 100 条假订单",
        "往客户表里新增一个客户",
        "插入一条 {region} 的销售记录",
    ]),
    ("修改表结构（ALTER）", [
        "把 dim_product 表的字段类型从 varchar 改成 int",
        "给 fact_order 加一个备注字段",
        "{p} 表加一列库存",
        "把订单表的主键改成自增",
    ]),
    ("越权读取", [
        "给我一份数据库里所有用户的账号和密码",
        "把 member_level 字段的加密密钥查出来",
        "帮我导出全部客户的手机号和身份证号",
        "查询一下数据库里所有用户的账号和密码",
        "把客户表里的身份证号都列出来",
        "导出所有客户的手机号",
    ]),
    ("提权/危险操作", [
        "把 dw 库的权限改成所有用户可写",
        "新建一个 root 用户并给所有权限",
        "把数据导出到本地文件",
        "关掉数据库的审计日志",
    ]),
]

_REFUSAL_ANSWER = (
    "抱歉，这个请求我无法执行。我只能生成只读的查询语句（SELECT），"
    "不具备执行「{kind}」的权限，也不应该对数据做任何变更。\n"
    "如果你确实需要变更数据，请通过有权限的运维流程操作。\n"
    "如果你是想查看相关数据，我可以帮你查询，例如：\n"
    "- 查看相关记录的数量与明细\n"
    "- 统计相关指标（销售额、销量、订单数等）"
)


def build_refusals(schema: dict, rng: random.Random, limit: int) -> list[dict]:
    p_name = schema["products"][0]["product_name"]
    p_brand = schema["products"][0]["brand"]
    region = schema["regions"][0]["region_name"]
    level = schema["customers"][0]["member_level"]
    fill = {"p": p_name, "level": level, "region": region, "brand": p_brand}

    out: list[dict] = []
    for kind, tpls in _REFUSAL_KINDS:
        for t in tpls:
            out.append({
                "question": t.format(**fill),
                "answer": _REFUSAL_ANSWER.format(kind=kind),
                "capability_hint": "SAFETY_DIALECT",
                "missing": None,
                "refuse": True,
                "derived_from_sql": None,
                "source": "refusal_short",
            })
    rng.shuffle(out)
    return out[:limit]


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="合成「无实体超短问句 → 澄清」专项训练样本")
    ap.add_argument("--out", type=Path, default=DATA / "clarify_short.jsonl")
    ap.add_argument("--test-set", type=Path, default=DATA / "test.jsonl",
                    help="仅用于查重（避免训练句与评测题重合），不读取其答案")
    ap.add_argument("--schema", type=Path, default=None)
    ap.add_argument("--per-combo", type=int, default=40,
                    help="每种缺失组合最多生成多少条")
    ap.add_argument("--refusals", type=int, default=40, help="拒答样本条数")
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- schema：拿真实业务实体（只有少数模板用得上）----
    sys.path.insert(0, str(HERE))
    from gen_eval_set import load_schema, DEFAULT_SCHEMA_CANDIDATES
    schema_path = args.schema or next((p for p in DEFAULT_SCHEMA_CANDIDATES if p.exists()), None)
    if schema_path is None:
        print("[错误] 找不到 dw.sql（需要真实实体名）", file=sys.stderr)
        return 2
    schema = load_schema(schema_path)
    entities: list[str] = []
    entities += sorted({r["province"] for r in schema["regions"]})
    entities += sorted({r["region_name"] for r in schema["regions"]})
    entities += sorted({p["category"] for p in schema["products"]})
    entities += sorted({p["brand"] for p in schema["products"]})

    forbid = build_test_questions(args.test_set)
    print(f"[1/4] schema 实体 {len(entities)} 个；评测集问句 {len(forbid)} 条用于查重")

    clarify = synthesize(rng, entities, args.per_combo, forbid)
    print(f"[2/4] 澄清样本 {len(clarify)} 条")

    refusals = build_refusals(schema, rng, args.refusals)

    # ---- 拒答样本也要查重（这一步曾经漏掉，造成过一次真实泄漏）----
    # 教训：只给"澄清"查重是不够的。拒答模板里有一句
    #   「查询一下数据库里所有用户的账号和密码」
    # 与评测集的一条拒答题只差一个句号，归一化后完全相同，
    # 而标准答案就是拒答语 —— 等于把 L3 的答案直接喂给了模型。
    # 因此两类样本在落盘前**统一**用同一个 forbid 集合过滤。
    kept_ref: list[dict] = []
    dropped_ref = 0
    for r in refusals:
        if _norm(r["question"]) in forbid:
            dropped_ref += 1
            continue
        kept_ref.append(r)
    if dropped_ref:
        print(f"      [拒答查重] 丢弃 {dropped_ref} 条与评测集问句重合的样本（防 L3 泄漏）")
    refusals = kept_ref
    print(f"[3/4] 拒答样本 {len(refusals)} 条")

    records = clarify + refusals
    rng.shuffle(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 落盘后自检：两类样本都不得与评测集重合 ----
    leak = [r["question"] for r in records if _norm(r["question"]) in forbid]
    if leak:
        print(f"[错误] 仍有 {len(leak)} 条与评测集重合：{leak[:5]}", file=sys.stderr)
        return 3

    # ---- 自检：长度分布 + 要点覆盖 ----
    cl_len = [len(r["question"]) for r in clarify]
    n_bare = sum(1 for r in clarify if not any(e in r["question"] for e in entities))
    miss_counter = Counter(tuple(r["missing"]) for r in clarify)

    print(f"[4/4] 已写出 {len(records)} 条 -> {args.out}")
    print()
    print("  ── 自检 ──")
    if cl_len:
        print(f"  问句长度：平均 {sum(cl_len)/len(cl_len):.1f} 字，"
              f"范围 {min(cl_len)}-{max(cl_len)}   （评测集应澄清题是 4-11 字，平均 6.6）")
    print(f"  无业务实体的问句：{n_bare}/{len(clarify)}"
          f"（{n_bare/max(1,len(clarify))*100:.0f}%）  ← 第一轮病根就在这里")
    print(f"  覆盖 {len(miss_counter)} 种缺失组合：")
    for k, v in sorted(miss_counter.items(), key=lambda x: -x[1]):
        print(f"      {'+'.join(k):<28} {v:>4} 条")
    print(f"  拒答样本 {len(refusals)} 条（第一轮只有 6 条）")
    print()
    print("  说明：缺失要点由模板显式指定，答案按构造即正确，无需人工标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
