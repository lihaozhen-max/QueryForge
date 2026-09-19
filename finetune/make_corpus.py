#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练语料生成器（仅生成「问题」，不含标准 SQL）

用途
----
微调需要 1000 条训练样本，但仓库里只有 3 个示例问题，没有真实用户问题库。
本脚本用「维度 × 指标 × 问法 × 查询型态」的组合，程序化产出一批中文问数问题，
作为 build_train_set.py 的输入，由现有 LangGraph 流水线去生成 SQL。

与评测集的区别（重要）
    test.jsonl 由模板直接给出**标准 SQL**，用于评测；
    本脚本只给**问题**，SQL 由流水线产出后再做执行校验，用于训练。
    两者走不同路径，避免"用流水线自己的输出考自己"。

用法
----
    python finetune/make_corpus.py                 # 默认产出 ~1200 条到 finetune/data/corpus.txt
    python finetune/make_corpus.py --out X.txt --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 复用评测集的解析逻辑，保证两边用的是同一套真实数仓取值
try:
    from gen_eval_set import load_schema, DEFAULT_SCHEMA_CANDIDATES
except ImportError:  # 允许从任意 cwd 运行
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_eval_set import load_schema, DEFAULT_SCHEMA_CANDIDATES


# --------------------------------------------------------------------------
# 问法模板：同一语义的多种中文表达，覆盖正式 / 口语 / 方言式说法
# --------------------------------------------------------------------------
def make_questions(schema: dict, rng: random.Random) -> list[str]:
    regions = schema["regions"]
    products = schema["products"]
    customers = schema["customers"]
    months = schema["months"]

    region_names = sorted({r["region_name"] for r in regions})
    provinces = sorted({r["province"] for r in regions})
    categories = sorted({p["category"] for p in products})
    brands = sorted({p["brand"] for p in products})
    levels = sorted({c["member_level"] for c in customers})
    genders = sorted({c["gender"] for c in customers})

    # 品牌 / 省份口语别名（3.2.2 字段值标准化的训练信号）
    brand_alias = {
        "苹果": ["苹果", "Apple"], "三星": ["三星", "Samsung"], "华为": ["华为", "Huawei"],
        "戴森": ["戴森", "Dyson"], "美的": ["美的", "Midea"], "耐克": ["耐克", "Nike"],
        "阿迪达斯": ["阿迪达斯", "Adidas"], "优衣库": ["优衣库", "Uniqlo"],
        "李维斯": ["李维斯", "Levi's"], "雀巢": ["雀巢", "Nestle"],
        "蒙牛": ["蒙牛"], "乐事": ["乐事", "Lays"], "奥利奥": ["奥利奥", "Oreo"],
        "亚马逊": ["亚马逊", "Amazon"], "Instant Pot": ["Instant Pot"],
    }
    province_alias = {p: [p, p.rstrip("省").rstrip("市")] for p in provinces}

    out: list[str] = []

    def emit(*variants: str) -> None:
        for v in variants:
            v = v.strip()
            if v:
                out.append(v)

    # ---------------- 全局单指标｜3.2.1 字段映射 + 3.2.3 指标口径 ----------------
    emit(
        "总销售额是多少？", "销售额总共多少？", "一共卖了多少钱？",
        "全部订单的成交总额是多少？", "营业额是多少？",
    )
    emit(
        "平均客单价是多少？", "客单价多少？", "平均每单多少钱？",
        "订单金额的平均值是多少？", "平均订单金额是多少？",
    )
    emit(
        "一共卖出了多少件商品？", "总销量是多少？", "累计卖了多少件？",
        "商品销售数量合计是多少？",
    )
    emit(
        "一共有多少个订单？", "订单总数是多少？", "总共成交了多少笔？",
    )
    emit(
        "一共有多少个客户下过单？", "下单客户数是多少？", "有多少个不同的客户买过东西？",
    )
    emit(
        "单笔最高的订单金额是多少？", "最大的一笔订单多少钱？", "金额最高的订单是多少钱？",
    )
    emit(
        "单笔最低的订单金额是多少？", "最小的一笔订单多少钱？",
    )
    emit(
        "一共有多少种商品？", "在售商品数量是多少？", "商品一共有多少个？",
    )
    emit(
        "覆盖了多少个省份？", "一共涉及多少个省份？",
    )

    # ---------------- 按大区 / 省份｜3.2.5 多表关联 ----------------
    for rn in region_names:
        emit(
            f"{rn}大区的销售额是多少？", f"{rn}的销售额是多少？",
            f"{rn}地区卖了多少金额？", f"{rn}大区一共做了多少营业额？",
        )
        emit(
            f"{rn}的销量是多少？", f"{rn}大区卖了多少件商品？",
        )
        emit(
            f"{rn}有多少个订单？", f"{rn}大区的订单数量是多少？",
        )
    for pv in provinces:
        for al in province_alias[pv]:
            emit(
                f"{al}的销售额是多少？", f"{al}卖了多少金额？",
            )
            emit(
                f"{al}的订单量是多少？", f"{al}有多少笔订单？",
            )

    # ---------------- 按品类 / 品牌 / 商品｜3.2.2 字段值标准化 ----------------
    for cat in categories:
        emit(
            f"{cat}的销售额是多少？", f"{cat}这个品类卖了多少钱？",
            f"{cat}品类的成交总额是多少？",
        )
        emit(
            f"{cat}的销量是多少？", f"{cat}一共卖了多少件？",
        )
        emit(
            f"{cat}的订单数是多少？", f"{cat}有多少个订单？",
        )
    for brand in brands:
        for al in brand_alias.get(brand, [brand]):
            emit(
                f"{al}的销售额是多少？", f"{al}卖了多少金额？",
                f"{al}这个品牌贡献了多少销售额？",
            )
            emit(
                f"{al}卖了多少件？", f"{al}的销量是多少？",
            )
            emit(
                f"{al}的客单价是多少？", f"{al}平均每单多少钱？",
            )
    for prod in products:
        short = prod["product_name"].split()[0]
        emit(
            f"{prod['product_name']}的销售额是多少？", f"{prod['product_name']}卖了多少？",
        )
        if short != prod["product_name"]:
            emit(
                f"{short}的销售额是多少？", f"{short}卖了多少金额？",
            )

    # ---------------- 按会员等级 / 性别｜3.2.5 ----------------
    for lv in levels:
        emit(
            f"{lv}会员的销售额是多少？", f"{lv}会员贡献了多少金额？",
            f"{lv}会员有多少人下过单？",
        )
        emit(
            f"{lv}会员的客单价是多少？", f"{lv}会员平均每单多少钱？",
        )
    for g in genders:
        emit(
            f"{g}性客户的销售额是多少？", f"{g}性客户的消费金额是多少？",
            f"{g}性客户有多少订单？",
        )

    # ---------------- 时间维度｜3.2.3 时间口径 ----------------
    month_alias = {1: ["1月", "一月份"], 2: ["2月", "二月份"], 3: ["3月", "三月份"]}
    for m in months:
        for label in month_alias.get(m, [f"{m}月"]):
            emit(
                f"{label}的销售额是多少？", f"{label}卖了多少钱？",
                f"{label}的销量是多少？",
            )
            emit(
                f"{label}的订单数是多少？", f"{label}有多少订单？",
            )
        emit(
            f"{m}月份的客单价是多少？", f"{m}月平均每单多少钱？",
        )
    emit(
        "第一季度的销售额是多少？", "Q1的销售额是多少？",
        "第一季度一共做了多少金额？", "第一季度的销量是多少？",
    )
    emit(
        "第一季度的客单价是多少？", "Q1平均每单多少钱？",
    )

    # ---------------- 分组聚合｜3.2.5 ----------------
    emit(
        "各个地区的销售额分别是多少？", "各大区的销售额排名是怎样的？",
        "按大区统计销售额", "每个大区的成交总额是多少？",
    )
    emit(
        "各省份的销售额排名是怎样的？", "各个省份的销售额分别是多少？",
        "按省份看销售额",
    )
    emit(
        "每个品类的销售额分别是多少？", "各品类的销售额排名如何？",
        "按品类统计成交金额",
    )
    emit(
        "各个品牌的销售额分别是多少？", "各品牌的销售额排名是怎样的？",
        "按品牌看销售额",
    )
    emit(
        "每个月的销售额分别是多少？", "各月销售额分别是多少？",
        "按月统计销售额", "月度销售额分别是多少？",
    )
    emit(
        "每个月的销量分别是多少？", "各月销量是多少？",
    )
    emit(
        "不同会员等级的销售额分别是多少？", "各会员等级贡献的销售额是多少？",
        "按会员等级统计销售额",
    )
    emit(
        "男性和女性的销售额分别是多少？", "不同性别的消费金额分别是多少？",
        "按性别统计销售额",
    )
    emit(
        "每个季度的销售额分别是多少？", "各季度销售额是多少？",
    )
    emit(
        "每个品类每个月的销售额分别是多少？", "各品类各月的成交金额是多少？",
        "品类和月份交叉的销售额",
    )
    emit(
        "各大区各品类的销售额分别是多少？", "按大区和品类交叉统计销售额",
    )

    # ---------------- TopN / 排序 / 过滤｜3.2.5 ----------------
    emit(
        "销售额最高的前5个商品是哪些？", "卖得最好的5个商品是什么？",
        "成交额排名前五的商品", "哪5个商品销售额最高？",
    )
    emit(
        "销量最高的前3个商品是哪些？", "卖得最多的3个商品是什么？",
    )
    emit(
        "销售额最高的3个大区是哪些？", "哪几个大区销售额排前三？",
    )
    emit(
        "消费金额最高的前5位客户是谁？", "消费最多的5个客户是哪些？",
    )
    emit(
        "销售额超过10000元的大区有哪些？", "哪些大区的销售额大于1万？",
    )
    emit(
        "销售额低于5000元的品牌有哪些？", "销售额不足5000的品牌是哪些？",
    )
    emit(
        "订单数超过10个的品类有哪些？", "哪些品类的订单数大于10？",
    )
    emit(
        "哪个大区的销售额最高？", "销售额最高的大区是哪个？",
        "哪个地区卖得最好？",
    )
    emit(
        "哪个品类的销量最高？", "销量最高的品类是哪个？",
    )
    emit(
        "哪个品牌的销售额最高？", "卖得最好的品牌是哪个？",
        "销售额排名第一的品牌是哪个？",
    )
    emit(
        "哪个商品的销售额最高？", "卖得最好的商品是什么？",
    )
    emit(
        "哪个月份的销售额最高？", "销售额最高的月份是哪个月？",
    )
    emit(
        "哪个会员等级的客单价最高？", "客单价最高的会员等级是哪个？",
    )
    emit(
        "各个大区的销售额从高到低排列", "把各大区销售额降序排列",
    )
    emit(
        "各个品类的订单数从多到少排列", "按订单数降序看品类",
    )

    # ---------------- 多条件组合｜3.2.5 ----------------
    for cat in categories[:5]:
        emit(
            f"{cat}1月的销售额是多少？", f"1月{cat}的销售额是多少？",
        )
        emit(
            f"{cat}2月的销量是多少？", f"2月{cat}卖了多少件？",
        )
    for rn in region_names[:5]:
        emit(
            f"{rn}1月的销售额是多少？", f"1月{rn}的销售额是多少？",
        )
        emit(
            f"{rn}的客单价是多少？", f"{rn}平均每单多少钱？",
        )
    for pv in provinces[:5]:
        for cat in categories[:3]:
            emit(f"{pv}的{cat}销售额是多少？")
    for rn in region_names[:4]:
        for cat in categories[:3]:
            emit(f"{rn}大区{cat}的销售额是多少？")
    for brand in brands[:6]:
        emit(f"{brand}1月的销售额是多少？")
        emit(f"{brand}2月的销量是多少？")

    # ---------------- 品牌 × 品类 交叉｜3.2.5（用真实配对，避免问不存在的货）----------------
    for cat in categories:
        cat_brands = sorted({p["brand"] for p in products if p["category"] == cat})
        for b in cat_brands:
            emit(
                f"{cat}里{b}的销售额是多少？", f"{b}在{cat}品类的销售额是多少？",
            )
            emit(
                f"{cat}里{b}卖了多少件？", f"{b}的{cat}产品销量是多少？",
            )

    # ---------------- 大区 × 品类 / 大区 × 月份｜3.2.5 三表 JOIN ----------------
    for rn in region_names:
        for cat in categories[:4]:
            emit(
                f"{rn}大区{cat}的销售额是多少？", f"{rn}的{cat}卖了多少金额？",
            )
            emit(
                f"{rn}大区{cat}的销量是多少？", f"{rn}的{cat}有多少订单？",
            )
    for pv in provinces:
        for m in months:
            emit(
                f"{pv}{m}月的销售额是多少？", f"{m}月{pv}卖了多少金额？",
            )
    for cat in categories:
        for m in months:
            emit(
                f"{cat}{m}月的销售额是多少？", f"{m}月{cat}卖了多少钱？",
            )
            emit(
                f"{cat}{m}月的订单数是多少？",
            )
    for b in brands[:8]:
        for m in months:
            emit(f"{b}{m}月的销售额是多少？")

    # ---------------- 会员等级 × 品类 / 性别 × 品类｜3.2.5 ----------------
    for lv in levels:
        for cat in categories[:4]:
            emit(f"{lv}会员买{cat}花了多少钱？")
        for m in months:
            emit(f"{lv}会员{m}月的销售额是多少？")
    for g in genders:
        for cat in categories[:4]:
            emit(f"{g}性客户买{cat}的金额是多少？")

    # ---------------- 商品级补充｜3.2.1 / 3.2.2 ----------------
    for prod in products:
        emit(
            f"商品{prod['product_name']}一共卖了多少件？",
            f"{prod['product_name']}的订单数是多少？",
            f"{prod['product_name']}的平均订单金额是多少？",
        )
    for brand in brands:
        emit(
            f"{brand}一共有多少个商品？",
            f"{brand}的订单数是多少？",
            f"{brand}卖出的商品件数合计是多少？",
        )

    # ---------------- TopN 扩充（不同 N、不同维度）----------------
    for n in (3, 5, 10):
        emit(
            f"销售额最高的前{n}个大区是哪些？",
            f"销售额最高的前{n}个品牌是哪些？",
            f"销售额最高的前{n}个品类是哪些？",
            f"销量最高的前{n}个省份是哪些？",
            f"订单数最多的前{n}个品类是哪些？",
            f"客单价最高的前{n}个品牌是哪些？",
        )
    emit(
        "销售额最高的前10个商品是哪些？", "消费金额最高的前10位客户是谁？",
    )

    # ---------------- 时间补充 ----------------
    emit(
        "1月到2月的销售额是多少？", "前两个月的销售额合计是多少？",
        "1月和3月的销售额分别是多少？",
    )
    emit(
        "3月每天卖得最好的商品是什么？",
    )

    # ---------------- 3.2.4 澄清类（信息不全，应反问而非硬猜）----------------
    clarify = [
        "看看销售情况", "销售怎么样？", "最近生意好不好", "帮我分析一下数据",
        "销售额是多少", "哪个卖得最好", "表现最好的地区", "客户情况怎么样",
        "帮我统计一下", "订单情况", "趋势如何", "对比一下各个渠道",
        "利润率是多少", "同环比增长多少", "客单价怎么样", "哪个品牌最受欢迎",
        "最近一段时间销售额多少", "按地区看一下", "数据怎么样", "业务表现如何",
        "给我看看关键指标", "哪个产品值得重点推", "经营状况好不好",
        "这段时间表现如何", "帮我做个报表", "有什么异常吗", "情况如何",
    ]
    for q in clarify:
        out.append(q)

    # ---------------- 3.2.6 安全类（危险/越权请求，应拒答）----------------
    safety = [
        "把 dim_product 表里下架的商品删掉。", "帮我把订单金额改成0。",
        "删除 fact_order 表。", "查询一下数据库里所有用户的账号和密码。",
        "把销量低于10的商品记录清理掉。", "更新一下客户表里的会员等级。",
        "帮我在订单表里插入一条测试数据。", "把这张表的字段类型改一下。",
        "DROP TABLE dim_region", "truncate fact_order",
        "查一下数据库的用户名和密码", "把昨天的订单都删了",
    ]
    for q in safety:
        out.append(q)

    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    rng.shuffle(uniq)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="生成训练用问题语料（不含标准 SQL）")
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "corpus.txt")
    parser.add_argument("--seed", type=int, default=20250920)
    parser.add_argument("--total", type=int, default=0, help="截断到 N 条，0 表示全部")
    args = parser.parse_args()

    schema_path = args.schema
    if schema_path is None:
        schema_path = next((p for p in DEFAULT_SCHEMA_CANDIDATES if p.exists()), None)
    if schema_path is None or not schema_path.exists():
        print("[错误] 找不到 dw.sql，请用 --schema 指定", file=sys.stderr)
        return 2

    schema = load_schema(schema_path)
    rng = random.Random(args.seed)
    questions = make_questions(schema, rng)

    if args.total > 0:
        questions = questions[: args.total]

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 便于用记事本/Excel 直接查看修改
    out_path.write_text("\n".join(questions) + "\n", encoding="utf-8")

    print(f"语料已写出：{out_path}")
    print(f"  问题总数：{len(questions)}")
    print(f"  数据来源：{schema_path}")
    print(f"  随机种子：{args.seed}（固定种子，可复现）")
    print()
    print("  说明：本文件只有问题，标准 SQL 由现有 LangGraph 流水线产出后执行校验。")
    print("       训练集与评测集走不同路径，避免自我考核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
