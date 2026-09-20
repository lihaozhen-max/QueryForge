import re

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt

"""
1. 利用大模型对table_infos进行过滤处理生成需要的字段名和表名的字典容器：
    {”表名1“：["字段名1", "字段名2"], ”表名2“：["字段名3", "字段名4"]}
2. 对table_infos中的表信息和字段信息进行过滤
3. 返回过滤后的table_infos

⚠️ 关于"补回 join 键"（2026-09-20 新增）
----------------------------------------
**现象**：线上真跑时报
    Unknown column 'fo.order_month' in 'on clause'
模型生成的 JOIN 条件里用了一个**根本不存在的字段**。

**根因**：过滤节点把连接两表所需的键列也裁掉了。
实测那次留下的 schema 是：
    dim_date   -> month
    fact_order -> order_amount
模型手上没有任何能连接这两列的键，于是**只能猜**，
先猜 order_date、再猜 order_month，两次都不存在。

**为什么之前没暴露**：评测时为了"只变一个变量"，预测走的是
`dump_context.py --fast`，**跳过了 filter_table / filter_metric 这两个节点**。
所以 87.1% 的 EX 是在"不过滤"的条件下拿到的，线上过滤后的问题一直没被测到。

**修法**：过滤之后做一次确定性补全 —— 凡是「多张入选表里同名出现」且
形如 `xxx_id` 的列（如 fact_order.date_id / dim_date.date_id），
说明它是连接键，被裁掉就补回来。补全只用**已经召回过的列对象**，
不查库、不调模型、不硬编码表结构，因此对 schema 变化免疫。
"""

# 「连接键」的命名特征：小写、以 _id 结尾、且不是纯 id
_JOIN_KEY_RE = re.compile(r"^(?!id$)[a-z][a-z0-9_]*_id$")


def _detect_join_keys(table_infos: list) -> dict[str, list]:
    """
    从**完整的** table_infos 里找出连接键。

    返回 {表名: [列对象, ...]}，只包含那些"在多张表里同名出现"的 *_id 列。
    为什么要求跨表同名：单表内也有 order_id 这种主键，它不用于连接，
    补回来只会让 schema 变长、干扰模型。真正的连接键必然在多张表里同时存在。
    """
    by_name: dict[str, list] = {}
    for t in table_infos:
        for c in t.get("columns") or []:
            if _JOIN_KEY_RE.match((c.get("name") or "").lower()):
                by_name.setdefault(c["name"], []).append((t["name"], c))

    out: dict[str, list] = {}
    for name, holders in by_name.items():
        if len({tn for tn, _ in holders}) < 2:
            continue          # 只出现在一张表 -> 不是连接键
        for tn, col in holders:
            out.setdefault(tn, []).append(col)
    return out


async def filter_table(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "过滤表信息"})

    try:
        query = state['query']
        table_infos = state['table_infos']
        # 过滤前先留一份完整快照：补 join 键时要从这里取列对象
        before = {
            t['name']: {c['name']: c for c in (t.get('columns') or [])}
            for t in table_infos
        }
        join_keys = _detect_join_keys(table_infos)

        # 1. 利用大模型对table_infos进行过滤处理生成需要的字段名和表名的字典容器：
        #     {”表名1“：["字段名1", "字段名2"], ”表名2“：["字段名3", "字段名4"]}
        prompt_template = PromptTemplate(
            template=load_prompt('filter_table_info'),
            input_variables=['query', 'table_infos']
        )

        output_parser = JsonOutputParser()
        chain = prompt_template | llm |output_parser
        result = chain.invoke({
            'query': query,
            'table_infos': yaml.dump(table_infos, allow_unicode=True, sort_keys=False)
        })

        for table in table_infos[:]:
            table_name = table['name']
            if (table_name not in result):
                table_infos.remove(table)
            else:
                for column in table['columns'][:]:
                    column_name = column['name']
                    if (column_name not in result[table_name]):
                        table['columns'].remove(column)

        # 2. 补回被裁掉的连接键（见模块 docstring 的说明）
        restored: list[str] = []
        for table in table_infos:
            tname = table['name']
            have = {c['name'] for c in (table.get('columns') or [])}
            for key_col in join_keys.get(tname, []):
                kname = key_col['name']
                if kname in have:
                    continue
                if len(table_infos) > 1:      # 单表查询不需要 join 键
                    full = before.get(tname, {}).get(kname)
                    if full is not None:
                        table['columns'].append(full)
                        have.add(kname)
                        restored.append(f"{tname}.{kname}")
        if restored:
            logger.info(f"补回被过滤掉的连接键：{restored}")

        logger.info(f'过滤表和字段完成： {table_infos}')

        return {'table_infos': table_infos}
    except Exception as e:
        logger.error(f"过滤表信息失败： {str(e)}")
        raise























