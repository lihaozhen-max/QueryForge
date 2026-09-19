# 负责定义FastAPI生命周期事件
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # print("应用启动启动时执行一次， 一般执行一些一次性初始化的操作，比如：初始化客户端")
    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    embedding_client_manager.init_client()
    yield
    # 关闭时执行的逻辑

    # print("应用结束前执行一次，一般执行一些收尾的工作，比如：关闭客户端")
    await dw_mysql_client_manager.close()
    await meta_mysql_client_manager.close()
    await es_client_manager.close()
    await qdrant_client_manager.close()
