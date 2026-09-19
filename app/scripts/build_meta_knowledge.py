"""
构建元数据库知识的同步脚本（入口）
"""
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

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
from app.services.meta_knowledge_service import MetaKnowledgeService

"""
1. 初始化客户端
2. 创建session对象
3. 创建持久层对象和业务对象
4. 调用构建的业务方法
5. 如果成功了，提交事务，如果失败了，回滚事务
6. 无论成功失败，最终都关闭客户端
"""

async def start_build():
    logger.info('开始构建元知识库')
    # 初始化客户端
    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    embedding_client_manager.init_client()

    try:
        assert dw_mysql_client_manager.session_factory
        async with (
            dw_mysql_client_manager.session_factory() as dw_session,
            meta_mysql_client_manager.session_factory() as meta_session
        ):
            meta_session: AsyncSession
            # 创建service对象

            service=MetaKnowledgeService(
                dw_mysql_repo=DWMysqlRepository(dw_session),
                meta_mysql_repo=MetaMysqlRepository(meta_session),
                value_es_repo=ValueESRepository(es_client_manager.client),
                column_qdrant_repo=ColumnQdrantRepository(qdrant_client_manager.client),
                metric_qdrant_repo=MetricQdrantRepository(qdrant_client_manager.client),
                embedding_client=embedding_client_manager.client
            )


            await service.build()

            await meta_session.commit()
            logger.info('构建元知识库完成')



    except Exception as e:
        logger.error(f'构建元知识库失败: {str(e)}')
        # 回滚事务
        await meta_session.rollback()
        raise
    finally:
        # 关闭所有客户端
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()
        await es_client_manager.close()
        await qdrant_client_manager.close()

if __name__ == '__main__':
    asyncio.run(start_build())



























