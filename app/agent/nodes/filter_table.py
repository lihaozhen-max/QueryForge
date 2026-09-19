import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from sqlalchemy.ext.asyncio import result

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
"""

async def filter_table(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "过滤表信息"})

    try:
        query = state['query']
        table_infos = state['table_infos']
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
        logger.info(f'过滤表和字段完成： {table_infos}')


        return {'table_infos': table_infos}
    except Exception as e:
        logger.error(f"过滤表信息失败： {str(e)}")
        raise























