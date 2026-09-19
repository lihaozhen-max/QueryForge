import uuid

import uvicorn
from fastapi import FastAPI,Request

from app.api.routers.query_router import query_router
from app.core.context import set_request_id
from app.core.lifespan import lifespan

# 创建应用对象
app = FastAPI(lifespan=lifespan)

# 注册路由器
app.include_router(query_router)

# 注册请求中间件 =》创建请求id并保存
@app.middleware("http")
async def middleware_func(request: Request, call_next):
    # 创建请求id并保存
    set_request_id(str(uuid.uuid4()))
    return await call_next(request)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)