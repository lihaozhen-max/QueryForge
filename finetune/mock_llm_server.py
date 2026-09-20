#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本机假模型服务：**OpenAI 兼容**，用来在没有 GPU 的时候联调整条流水线。

它不加载任何模型，只按规则编造回答：
  * 问数类问题 -> 返回一条语法正确的 MySQL SELECT（基于本项目的真实表结构）
  * 信息不全的短问题 -> 返回澄清话术（和微调模型的行为一致）

为什么需要它
------------
微调模型的真实服务要跑在租来的 GPU 上，而租卡是按小时计费的。
接入改造（提示词、节点、SSE、澄清分支）全是本机就能做的活，
用这个假服务就能把整条链路验完 —— **调试期不烧卡钱**。
等到要验真模型时，只要把 `LLM_BASE_URL` 换成隧道地址即可，其余代码零改动。

它也是**可复现的测试替身**：行为确定，适合写自动化验证。

用法
----
    python finetune/mock_llm_server.py --port 8000

然后本机项目里配：
    $env:LLM_PROVIDER='openai_compatible'
    $env:LLM_BASE_URL='http://127.0.0.1:8000/v1'
    $env:LLM_MODEL_NAME='qwen3-8b-lora'
    $env:LLM_API_KEY='EMPTY'

与真服务的差别（联调时需知道的边界）
------------------------------------
* 它**不会真的理解**问题，只做关键词匹配。所以它能验证"链路通不通、
  参数传得对不对、SSE 格式对不对"，但**不能验证模型效果**。
* 它**同时**响应两类任务：问数（SQL）与扩词/裁剪（JSON）。
  真微调模型在 JSON 类节点上的表现仍待实测。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# fastapi 导入必须在模块顶层，原因见 finetune/serve_qwen.py 的注释
# （`from __future__ import annotations` + 函数内导入 = Request 被当成查询参数 -> 422）
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

SERVED_NAME = "mock-qwen3-8b-lora"


# ==========================================================================
# 编造回答
# ==========================================================================
# 表结构事实（与本项目 dw 库一致），用于生成语法正确的 SQL
_SQL_RULES = [
    # (匹配词, 生成的 SQL)
    (("销售额", "gmv", "成交额", "营收"),
     "SELECT SUM(f.order_amount) AS gmv\n"
     "FROM fact_order f\n"
     "JOIN dim_date d ON f.date_id = d.date_id\n"
     "WHERE d.month = {month}"),
    (("销量", "件数", "数量", "卖了多少"),
     "SELECT SUM(f.order_quantity) AS total_qty\n"
     "FROM fact_order f\n"
     "JOIN dim_date d ON f.date_id = d.date_id\n"
     "WHERE d.month = {month}"),
    (("客单价", "aov", "平均订单"),
     "SELECT AVG(f.order_amount) AS aov\n"
     "FROM fact_order f\n"
     "JOIN dim_date d ON f.date_id = d.date_id\n"
     "WHERE d.month = {month}"),
    (("订单数", "订单量", "多少单", "订单数量"),
     "SELECT COUNT(*) AS order_cnt\n"
     "FROM fact_order f\n"
     "JOIN dim_date d ON f.date_id = d.date_id\n"
     "WHERE d.month = {month}"),
]

# 省份 / 大区 / 品类 / 品牌（用于拼 WHERE）
_PROVINCES = ["广东省", "浙江省", "四川省", "北京市", "上海市", "湖北省",
              "广东", "浙江", "四川", "北京", "上海", "湖北"]
_REGIONS = ["华南", "华东", "西南", "华北", "华中"]
_CATEGORIES = ["手机数码", "家用电器", "鞋靴", "服饰", "食品饮料", "休闲零食"]
_BRANDS = ["苹果", "华为", "三星", "戴森", "美的", "耐克", "阿迪达斯",
           "优衣库", "李维斯", "雀巢", "蒙牛", "乐事", "奥利奥", "iPhone", "Galaxy"]

_MONTH_RE = re.compile(r"(\d{1,2})\s*月")
_YEAR_RE = re.compile(r"(20\d{2})\s*年")


def _pick_month(q: str) -> int | None:
    m = _MONTH_RE.search(q)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 3:      # dw.dim_date 只覆盖 2025 Q1
            return v
    if "一" in q and "月" in q:
        return None
    for word, v in (("一月", 1), ("二月", 2), ("三月", 3), ("一季度", None), ("Q1", None)):
        if word in q:
            return v
    return None


def _extra_where(q: str) -> list[str]:
    """从问题里抽出实体条件，保证生成的 SQL 真能查到数据。"""
    out = []
    for prov in sorted(_PROVINCES, key=len, reverse=True):
        if prov in q:
            base = prov[:-1] if prov.endswith(("省", "市")) else prov
            out.append(f"dr.province = '{base}省'" if base in ("广东", "浙江", "四川", "湖北")
                       else f"dr.province = '{base}市'")
            break
    for region in _REGIONS:
        if region in q:
            out.append(f"dr.region_name = '{region}'")
            break
    for cat in sorted(_CATEGORIES, key=len, reverse=True):
        if cat in q:
            out.append(f"dp.category = '{cat}'")
            break
    for br in sorted(_BRANDS, key=len, reverse=True):
        if br in q:
            out.append(f"dp.brand = '{br}'")
            break
    return out


def _needs_join(q: str) -> tuple[bool, bool]:
    """返回 (要不要 join dim_region, 要不要 join dim_product)"""
    need_r = any(x in q for x in _PROVINCES + _REGIONS)
    need_p = any(x in q for x in _CATEGORIES + _BRANDS)
    return need_r, need_p


# 信息不全 -> 澄清（与微调模型学到的行为对齐）
_CLARIFY_PATTERNS = [
    (("销售额是多少", "销量是多少", "利润是多少", "客单价是多少", "订单数是多少"),
     ["时间范围"]),
    (("哪个卖得最好", "哪个品牌最受欢迎", "利润率是多少", "哪个最好"),
     ["指标口径"]),
    (("销售怎么样", "生意好不好", "经营得怎么样", "情况怎么样", "看看销售情况", "订单情况"),
     ["时间范围", "指标口径"]),
    (("按地区", "按品类", "分渠道", "按品牌统计", "分组看"),
     ["分析维度", "指标口径", "时间范围"]),
    (("趋势如何", "走势如何", "趋势怎么样"),
     ["分析对象", "时间范围"]),
]

_ASK = {
    "时间范围": "统计的时间范围？（例如：1月 / 第一季度 / 全部数据）",
    "统计粒度": "希望按什么维度看？（例如：按天 / 按月 / 按大区 / 按品类）",
    "指标口径": "具体看哪个指标？（例如：销售额 GMV / 客单价 AOV / 销量 / 订单数）",
    "分析对象": "想分析的对象是什么？（例如：销售业绩 / 客户 / 商品 / 地区）",
    "分析维度": "想按哪个维度拆分看？（例如：大区 / 省份 / 品类 / 品牌 / 月份）",
}


def _clarify(missing: list[str]) -> str:
    lines = ["这个问题还缺少一些必要信息，我需要确认后才能给出准确结果："]
    for i, m in enumerate(missing, 1):
        lines.append(f"{i}. {_ASK.get(m, m + '？')}")
    return "\n".join(lines)


def _gen_sql(q: str) -> str:
    month = _pick_month(q)
    if month is None:
        # 没给可识别的时间 -> 与模型一致：反问
        return _clarify(["时间范围"])

    for keys, tpl in _SQL_RULES:
        if any(k in q.lower() or k in q for k in keys):
            sql = tpl.format(month=month)
            break
    else:
        sql = ("SELECT SUM(f.order_amount) AS gmv\n"
               "FROM fact_order f\n"
               "JOIN dim_date d ON f.date_id = d.date_id\n"
               "WHERE d.month = {month}").format(month=month)

    need_r, need_p = _needs_join(q)
    joins, wheres = [], []
    if need_p:
        joins.append("JOIN dim_product dp ON f.product_id = dp.product_id")
    if need_r:
        joins.append("JOIN dim_region dr ON f.region_id = dr.region_id")
    wheres += _extra_where(q)

    if joins:
        # 插到 FROM ... 之后、WHERE 之前
        head, _, tail = sql.partition("WHERE")
        sql = head.rstrip() + "\n" + "\n".join(joins) + "\nWHERE" + tail
    if wheres:
        sql = sql.rstrip() + " AND " + " AND ".join(wheres)
    return sql


def _gen_json(q: str) -> str:
    """扩词 / 裁剪类节点要 JSON。真微调模型在这类任务上表现待测，这里给个合理形状。"""
    # 裁剪表：返回传入表名的一部分
    names = re.findall(r"name:\s*([A-Za-z_][\w]*)", q)
    uniq = list(dict.fromkeys(names))
    if "filter" in q.lower() or "筛选" in q:
        return json.dumps(uniq[:3] or ["fact_order"], ensure_ascii=False)
    # 扩词：给几个业务词
    return json.dumps(["销售额", "订单量", "客户", "商品", "大区"], ensure_ascii=False)


def make_answer(user_content: str) -> str:
    q = user_content.strip()
    # 判断任务类型：prompt 里出现这些词的就是非 SQL 类节点
    if ("关键词" in q or "扩展" in q or "筛选" in q or "过滤" in q) and "用户查询" not in q[-60:]:
        return _gen_json(q)

    # 问数类：先看要不要澄清
    tail = q[-120:]        # 只匹配【用户查询】附近，避免被 schema 里的词误触发
    for keys, missing in _CLARIFY_PATTERNS:
        if any(k in tail for k in keys):
            return _clarify(missing)
    return _gen_sql(tail)


# ==========================================================================
# HTTP
# ==========================================================================
def build_app():
    app = FastAPI(title="Mock LLM (OpenAI compatible)")

    @app.get("/health")
    def health():
        return {"status": "ok", "mock": True, "model": SERVED_NAME}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": SERVED_NAME, "object": "model",
                          "created": int(time.time()), "owned_by": "mock"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            return JSONResponse({"error": {"message": "messages 不能为空",
                                           "type": "invalid_request_error"}}, status_code=400)

        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content") or ""
                break

        content = make_answer(user)
        n_in = max(1, len(user) // 2)
        n_out = max(1, len(content) // 2)

        # 打印一条，便于确认请求真的到过这里、以及模型看到了什么
        print(f"[mock] <- {len(user):>5} 字符 | 输出 {len(content):>4} 字符 | "
              f"{content[:60]!r}", flush=True)

        if not body.get("stream"):
            return JSONResponse({
                "id": f"chatcmpl-mock{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SERVED_NAME,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                          "total_tokens": n_in + n_out},
            })

        cid = f"chatcmpl-mock{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        async def gen():
            yield ("data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]}, ensure_ascii=False) + "\n\n")
            if content:
                yield ("data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": SERVED_NAME,
                    "choices": [{"index": 0, "delta": {"content": content},
                                 "finish_reason": None}]}, ensure_ascii=False) + "\n\n")
            yield ("data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                          "total_tokens": n_in + n_out}}, ensure_ascii=False) + "\n\n")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="本机假模型服务（OpenAI 兼容）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn
    print(f"[mock] 监听 http://{args.host}:{args.port}   model={SERVED_NAME}", flush=True)
    print("[mock] 它不加载模型，只按关键词编造回答，用于本机联调（不烧 GPU）", flush=True)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
