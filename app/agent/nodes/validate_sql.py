from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger

async def validate_sql(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer("校验SQL")

    sql=''
    try:
        sql  = state['sql']

        dw_mysql_repo = runtime.context['dw_mysql_repo']

        # 检查SQL语法是否合法
        await dw_mysql_repo.validate_sql(sql)

        logger.info(f'校验SQL成功: {sql}')

        return {'error': None}
    except Exception as e:
        # 本次校验失败, 校正次数+1; 用于限制"校验-校正"环的最大轮数, 避免死循环
        correct_sql_count = state.get('correct_sql_count', 0) + 1
        logger.error(f'校验SQL失败(第{correct_sql_count}次): {sql} - {str(e)}')
        return {'error': f'SQL语法错误: {str(e)}', 'correct_sql_count': correct_sql_count}
