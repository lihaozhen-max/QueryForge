import asyncio

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.agent.nodes.add_extra_context import add_extra_context
from app.agent.nodes.correct_sql import correct_sql
from app.agent.nodes.execute_sql import execute_sql
from app.agent.nodes.extract_keywords import extract_keywords
from app.agent.nodes.filter_metric import filter_metric
from app.agent.nodes.filter_table import filter_table
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.merge_retrieved_info import merge_retrieved_info
from app.agent.nodes.recall_column import recall_column
from app.agent.nodes.recall_metric import recall_metric
from app.agent.nodes.recall_value import recall_value
from app.agent.nodes.validate_sql import validate_sql
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.core.log import logger
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository

state_graph = StateGraph(
    state_schema=DataAgentState,
    context_schema=DataAgentContext
)

# 添加节点
state_graph.add_node("extract_keywords", extract_keywords)
state_graph.add_node("recall_column",recall_column)
state_graph.add_node("recall_metric",recall_metric)
state_graph.add_node("recall_value",recall_value)
state_graph.add_node("merge_retrieved_info",merge_retrieved_info)
state_graph.add_node("filter_metric",filter_metric)
state_graph.add_node("filter_table",filter_table)
state_graph.add_node("add_extra_context",add_extra_context)
state_graph.add_node("generate_sql",generate_sql)
state_graph.add_node("validate_sql",validate_sql)
state_graph.add_node("correct_sql",correct_sql)
state_graph.add_node("execute_sql",execute_sql)


# 添加边
state_graph.add_edge(START, "extract_keywords")
state_graph.add_edge("extract_keywords","recall_column")
state_graph.add_edge("extract_keywords","recall_metric")
state_graph.add_edge("extract_keywords","recall_value")
state_graph.add_edge("recall_column","merge_retrieved_info")
state_graph.add_edge("recall_metric","merge_retrieved_info")
state_graph.add_edge("recall_value","merge_retrieved_info")
state_graph.add_edge("merge_retrieved_info","filter_metric")
state_graph.add_edge("merge_retrieved_info","filter_table")
state_graph.add_edge("filter_metric","add_extra_context")
state_graph.add_edge("filter_table","add_extra_context")
state_graph.add_edge("add_extra_context","generate_sql")
state_graph.add_edge("generate_sql","validate_sql")

# "校验-校正"环的最大轮数: 校验失败后被允许去校正的最大次数, 超过则直接执行(由execute_sql抛出最终错误)
MAX_CORRECT_ATTEMPTS = 2

def _route_after_validate(state: DataAgentState) -> str:
    """校验通过直接执行; 校验失败且还没用满校正次数则去校正; 否则放弃校正直接执行"""
    if state.get('error') is None:
        return 'execute_sql'
    if state.get('correct_sql_count', 0) <= MAX_CORRECT_ATTEMPTS:
        return 'correct_sql'
    logger.warning(f"已校正{state.get('correct_sql_count')}次仍校验失败, 放弃校正直接执行")
    return 'execute_sql'

state_graph.add_conditional_edges("validate_sql",
                                  _route_after_validate,
                                {'execute_sql': 'execute_sql', 'correct_sql': 'correct_sql'}
                                  )

# 校正后回到校验节点复验, 确认校正结果真的可用, 而不是盲目执行
state_graph.add_edge('correct_sql', 'validate_sql')
state_graph.add_edge('execute_sql',END)
compiled_graph = state_graph.compile()




if __name__ == '__main__':
    # print(compiled_graph.get_graph().draw_ascii())
    # print(compiled_graph.get_graph().draw_mermaid())

    async def test():

        # 初始化客户端
        dw_mysql_client_manager.init_client()
        meta_mysql_client_manager.init_client()
        es_client_manager.init_client()
        qdrant_client_manager.init_client()
        embedding_client_manager.init_client()

        try:
            # 创建session对象
            assert dw_mysql_client_manager.session_factory
            async with (
                dw_mysql_client_manager.session_factory() as dw_session,
                meta_mysql_client_manager.session_factory() as meta_session
            ):
                meta_session: AsyncSession

                # 创建状态对象
                '''
                - 华北地区销售总额
                - 2025年各地区平均销售额
                - 各个地区iPhone去年卖了多少钱
                '''
                state = DataAgentState(query="各个地区iPhone去年卖了多少钱", correct_sql_count=0)
                # 创建上下文件对象
                context = DataAgentContext(
                    dw_mysql_repo=DWMysqlRepository(dw_session),
                    meta_mysql_repo=MetaMysqlRepository(meta_session),
                    value_es_repo=ValueESRepository(es_client_manager.client),
                    column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
                    metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
                    embedding_client=embedding_client_manager.client
                )

                # 异步流式执行图
                async for chunk in compiled_graph.astream(
                        input=state,
                        context=context,
                        stream_mode="custom"  # 自定义节点输出
                ):
                    print("----", chunk)
        except Exception as e:
            logger.error(f"搜索失败：{str(e)}")
            raise
        finally:
            # 关闭所有客户端
            await dw_mysql_client_manager.close()
            await meta_mysql_client_manager.close()
            await es_client_manager.close()
            await qdrant_client_manager.close()


    asyncio.run(test())

