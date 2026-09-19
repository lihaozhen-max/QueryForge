from langgraph.runtime import Runtime

from datetime import datetime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, DateInfoState, DBInfoState
from app.core.log import logger

async def add_extra_context(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "添加额外信息"})

    try:
        dw_mysql_repo = runtime.context["dw_mysql_repo"]

        today = datetime.today()
        date = today.strftime("%Y-%m-%d")
        weekday = today.strftime("%A")
        quarter = f"Q{(today.month+2)//3}"
        date_info = DateInfoState(
            date=date, weekday=weekday, quarter=quarter
        )

        db_info_dict = await dw_mysql_repo.get_db_info()
        db_info = DBInfoState(**db_info_dict)

        logger.info(f'添加额外信息: date_info={date_info}, db_info={db_info}')



        return {'date_info': date_info, 'db_info': db_info}
    except Exception as e:
        logger.error(f"添加额外信息失败： {str(e)}")
        raise
