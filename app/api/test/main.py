# 技术点1: 处理不同类型的请求
# 技术点2：获取不同类型的参数
# 技术点3：使用路由器
# 技术点4：SSE流式响应
# 技术点5：生命周期
# 技术点6：中间件
# 技术点7：依赖注入

import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.params import Depends
from pydantic import  BaseModel
from starlette.responses import StreamingResponse

from app.api.test.order_router import order_router
from app.api.test.product_router import product_router
from app.clients.es_client_manager import es_client_manager
from app.models.es.value_info_es import ValueInfoES
from app.repositories.es.value_es_repository import ValueESRepository

# 技术点5: 生命周期
# 回调函数: 你定义, 你没有直接调用,但ta在特定时机执行
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('应用启动时执行一次,一般执行一些一次性初始化的操作,比如:初始化客户端')
    es_client_manager.init_client()
    yield
    print('应用结束前执行一次, 一般执行一些收尾工作,比如:关闭客户端')
    await es_client_manager.close()

app = FastAPI(lifespan=lifespan)

# 技术点7：依赖注入
# 路由函数依赖（需要使用的）的模块/对象通过路由函数的参数注入进来
# depend inject   DI

# 根据keyword参数，查询ES库中匹配的字段值信息列表   keyword=华北地区
# 当前路由函数需要使用ValueESRepository对象(依赖对象)
# 需要使用DI

# 返回/产生依赖对象的函数  => 每次处理请求都会执行

def get_value_es_repo():
    print('---get_value_es_repo()')
    return  ValueESRepository(es_client_manager.client)

@app.post('/di')
async def test_di(keyword:str, value_es_repo: ValueESRepository=Depends(get_value_es_repo)):
    print('处理对/di的post请求')
    value_infos: list[ValueInfoES] = await value_es_repo.search(keyword)
    return {'data': value_infos}


# 技术6: 中间件
@app.middleware('http')
async def middleware_func(request: Request, call_next):
    print('在路由函数处理请求前执行')# 技术点7：依赖注入
# 路由函数依赖（需要使用的）的模块/对象通过路由函数的参数注入进来
# depend inject   DI

# 根据keyword参数，查询ES库中匹配的字段值信息列表   keyword=华北地区
# 当前路由函数需要使用ValueESRepository对象(依赖对象)
# 需要使用DI

# 返回/产生依赖对象的函数  => 每次处理请求都会执行
    response = await call_next(request)
    print('在路由函数处理请求后执行')

    return response
























app = FastAPI()

# 技术点4：SSE流式响应
async  def fake_stream():
    for i in range(10):
        yield f'data: {{"message": "abc{i}"}} \n\n'
        await asyncio.sleep(1)

async def call_async_generator():
    async for chunk in fake_stream():
        print('---', chunk)

@app.get('/sse_stream')
def test_sse_stream():
    return StreamingResponse(fake_stream(), media_type="text/event-stream")




# 技术点3：使用路由器
# 订单模块 5
# 商品模块 6
# 需要对接口按功能模块进行拆分 =》使用路由器来管理各个模块自己的路由
app.include_router(product_router, prefix="/v1")
app.include_router(order_router, prefix="/v2")


# 技术点2：获取不同类型的参数

"""
- 3种携带文本参数的方式
  - query参数：请求路径？后面的参数，如：/xxx?name=tom&age=12
  - path参数：看似像路径，与路由路径占位对应的部分，如：路由路径：/xxx/{id}，请求：/xxx/2
  - body参数：请求体参数，一般是json格式，如：{"name": "tom", "age": 12}
- 路由函数接收3种不同的参数
  - 接收 body参数：BaseModel子类型的形参
  - 接收path参数：与路由路径中占位同名的形参
  - 接收query参数：其它形参
"""

class BodyParams(BaseModel):
    name: str
    age: int

@app.post('/params/{id}')
def test_params(body_params: BodyParams, id:int, gender:str):
    print('处理/params的post请求', body_params.name, body_params.age, id, gender)

    return {'id': id, 'name': body_params.name, 'age': body_params.age, gender: gender}

# 技术点1 处理不同类型的请求
@app.get('/xxx')
def test_get():
    print('处理对/xxx的GET请求')
    return {'message': '处理get请求的数据结果...'}

if __name__== '__main__':
    uvicorn.run(app, host='0.0.0.0',port=8000)
    # asyncio.run(call_async_generator())










