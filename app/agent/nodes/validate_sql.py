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
        logger.error(f"校验SQL失败： {sql}")
        return {'error': f'SQL语法错误: {str(e)}'}






    except Exception as e:
        logger.error(f"校验SQL失败： {str(e)}")
        return {"error": f"校验SQL失败： {str(e)}"}
