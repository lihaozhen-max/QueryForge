from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.prompt.prompt_loader import load_prompt

"""
1. 利用大模型对提问进行语义化的关键词提取，与jieba提取关键词进行去重合并=>keywords
2. 对keywords进行批量向量化 =》 keyword_vectors
3. 对每个keyword_vector, 去qdrant搜索匹配的字段信息列表 =》list[ColumnInfoQdrant]
4. 将得到的所有字段信息列表进行去重合并 =》recall_columns: list[ColumnInfoQdrant]
5. 返回recall_columns数据
"""
async def recall_column(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "召回字段"})

    try:

        query = state['query']
        keywords = state['keywords']
        column_qdrant_repo = runtime.context['column_qdrant_repo']
        embedding_client = runtime.context['embedding_client']

        # 利用大模型对提问进行语义化的关键词提取,与jieba提取关键词进行去重合并 => keywords

        prompt_template = PromptTemplate(
            template=load_prompt('extend_keywords_for_column_recall'),
            input_variables=["query"]
        )
        output_parser = JsonOutputParser()
        chain = prompt_template | llm | output_parser
        result = chain.invoke({'query': query})
        logger.info(f'recall_column llm keywords={result}')
        keywords = list(set(result + keywords))

        # 对keywords进行批量向量化
        keyword_vectors = await embedding_client.aembed_documents(keywords)
        # 对每个keyword_vector, 去qdrant搜索匹配的字段信息列表
        column_infos_dict: dict[str, ColumnInfoQdrant] = {}
        for keyword_vector in keyword_vectors:
            column_infos:  list[ColumnInfoQdrant] = await column_qdrant_repo.search(keyword_vector)
            # 将得到的所有字段信息列表进行去重合并
            for column_info in column_infos:
                column_id = column_info['id']

                if column_id not in column_infos_dict:
                    column_infos_dict[column_id] = column_info
        recall_columns :list[ColumnInfoQdrant] = list(column_infos_dict.values())
        logger.info(f'召回字段完成: {recall_columns}')

        # 返回recall_columns数据
        return {"recall_columns": recall_columns}
    except Exception as e:
        logger.error(f'召回字段失败： {str(e)}')
        raise











 






















