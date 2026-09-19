
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.prompt.prompt_loader import load_prompt

"""
1. 利用大模型对提问进行语义化的关键词提取，与jieba提取关键词进行去重合并=>keywords
2. 对keywords进行批量向量化 =》 keyword_vectors
3. 对每个keyword_vector, 去qdrant搜索匹配的指标信息列表 =》list[ColumnInfoQdrant]
4. 将得到的所有指标信息列表进行去重合并 =》recall_metrics: list[MetricInfoQdrant]
5. 返回recall_metrics数据
"""
async def recall_metric(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "召回指标"})

    try:

        query = state["query"]
        keywords = state["keywords"]
        metric_qdrant_repo = runtime.context["metric_qdrant_repo"]
        embedding_client = runtime.context["embedding_client"]

        # 1. 利用大模型对提问进行语义化的关键词提取，与jieba提取关键词进行去重合并=>keywords
        prompt_template = PromptTemplate(
            template=load_prompt("extend_keywords_for_metric_recall"),
            input_variables=["query"]
        )
        output_parser = JsonOutputParser()
        chain = prompt_template | llm | output_parser
        result = chain.invoke({"query": query})   # list[keyword, keyword]
        logger.info(f"recall_metric llm keywords={result}")
        keywords = list(set(result + keywords))

        # 2. 对keywords进行批量向量化 =》 keyword_vectors
        keyword_vectors = await embedding_client.aembed_documents(keywords)
        # 3. 对每个keyword_vector, 去qdrant搜索匹配的指标信息列表 =》list[MetricInfoQdrant]
        metric_infos_dict: dict[str, MetricInfoQdrant] = {}
        for keyword_vector in keyword_vectors:
            metric_infos: list[MetricInfoQdrant] = await metric_qdrant_repo.search(keyword_vector)
            # 4. 将得到的所有指标信息列表进行去重合并 =》recall_metrics: list[MetricInfoQdrant]
            for metric_info in metric_infos:
                metric_id = metric_info["id"]
                if metric_id not in metric_infos_dict:
                    metric_infos_dict[metric_id] = metric_info
        recall_metrics: list[MetricInfoQdrant] = list(metric_infos_dict.values())
        logger.info(f"召回指标完成： {recall_metrics}")

        # 5. 返回recall_metrics数据
        return {"recall_metrics": recall_metrics}
    except Exception as e:
        logger.error(f"召回指标失败： {str(e)}")
        raise
