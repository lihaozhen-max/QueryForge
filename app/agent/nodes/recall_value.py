from langchain_core import output_parsers
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.models.es.value_info_es import ValueInfoES
from app.prompt.prompt_loader import load_prompt


"""
1. 利用大模型对提问进行语义化的关键词提取，与jieba提取关键词进行去重合并=>keywords
2. 对每个keyword, 去es搜索匹配的字段值信息列表 =》list[ValueInfoES]
4. 将得到的所有字段值信息列表进行去重合并 =》recall_values: list[ValueInfoES]
5. 返回recall_values数据
"""


async def recall_value(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "校验SQL"})

    try:
        query = state['query']
        keywords = state['keywords']
        value_es_repo = runtime.context['value_es_repo']

        prompt_template = PromptTemplate(
            template=load_prompt('extend_keywords_for_value_recall'),
            input_variables=['query']
        )

        output_parser = JsonOutputParser()
        chain = prompt_template | llm | output_parser
        result = chain.invoke({"query": query})
        logger.info(f"recall_value llm keywords = {result}")
        keywords = list(set(result + keywords))

        value_infos_dict: dict[str, ValueInfoES] = {}
        for keyword in keywords:
            value_infos: list[ValueInfoES] = await value_es_repo.search(keyword)

            for value_info in value_infos:
                value_id = value_info['id']
                if value_id not in value_infos_dict:
                    value_infos_dict[value_id] = value_info
        recall_values: list[ValueInfoES] = list(value_infos_dict.values())

        logger.info(f'召回字段值完成： {recall_values}')

        return  {'recall_values': recall_values}











    except Exception as e:
        logger.error(f"召回字段值失败： {str(e)}")
        raise


















