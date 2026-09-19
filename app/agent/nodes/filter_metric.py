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
1. 利用大模型对metric_infos进行过滤处理生成需要的指标名的列表：
    ["指标名1", "指标名2"]
2. 对metric_infos中进行过滤
3. 返回过滤后的metric_infos
"""
async def filter_metric(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "过滤指标信息"})

    try:
        query = state["query"]
        metric_infos = state["metric_infos"]

        # 1. 利用大模型对metric_infos进行过滤处理生成需要的指标名的列表：
        #     ["指标名1", "指标名2"]
        prompt_template = PromptTemplate(
            template=load_prompt("filter_metric_info"),
            input_variables=["query", "metric_infos"]
        )
        output_parser = JsonOutputParser()
        chain = prompt_template | llm | output_parser
        result = chain.invoke({
            "query": query,
            "metric_infos": yaml.dump(metric_infos, allow_unicode=True, sort_keys=False)
        })
        # result ["指标名1", "指标名2"]
        # 2. 对metric_infos进行过滤
        # 问题：在遍历数组过程直接删除数组中的元素是有问题的 =》 对数组进行浅拷贝，遍历拷贝的数组
        for metric in metric_infos[:]:  # 浅拷贝
            metric_name = metric["name"]
            if (metric_name not in result):
                metric_infos.remove(metric)  # 过滤指标


        logger.info(f"过滤指标完成： {metric_infos}")

        # 3. 返回过滤后的metric_infos
        return {"metric_infos": metric_infos}
    except Exception as e:
        logger.error(f"过滤指标信息失败： {str(e)}")
        raise
