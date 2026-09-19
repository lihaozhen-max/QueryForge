# 负责定义查询接口
from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QuerySchema
from app.services.query_service import QueryService

query_router = APIRouter()
# 注册路由
# 使用async def: 该接口返回的是异步生成器驱动的SSE流, 同步def会被FastAPI放进线程池执行, 不必要地占用线程
@query_router.post('/api/query')
async def query(body_params: QuerySchema, query_service: QueryService = Depends(get_query_service)):
    return StreamingResponse(query_service.search(body_params.query),media_type='text/event-stream')


