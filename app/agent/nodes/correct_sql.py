import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    """修正SQL的节点"""
    # 获取流对象
    writer = runtime.stream_writer
    # 输出进度信息
    writer({'stage': '校正SQL'})
    try:
        # 获取sql
        sql = state['sql']
        error = state['error']
        query = state['query']
        table_infos = state['table_infos']
        metric_infos = state['metric_infos']
        date_info = state['date_info']
        db_info = state['db_info']

        prompt = PromptTemplate(template=load_prompt('correct_sql'),
                                input_variables=["sql", "error", "query", "table_infos", "metric_infos", "date_info", "db_info"])
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        sql = await  chain.ainvoke(
            {
                'query': query,
                'table_infos': yaml.dump(table_infos, allow_unicode=True,sort_keys=False),
                'metric_infos': yaml.dump(metric_infos, allow_unicode=True,sort_keys=False),
                'date_info': yaml.dump(date_info, allow_unicode=True,sort_keys=False),
                'db_info': yaml.dump(db_info, allow_unicode=True,sort_keys=False),
                'sql': sql,
                'error': error
            }
        )

        logger.info(f'校正sql成功： {sql}')
        return {'sql': sql}

    except Exception as e:

        logger.error(f'校正sql失败: {sql}')
        raise
