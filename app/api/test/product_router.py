from fastapi import  APIRouter
# 管理商品模的多个路由的路由器
product_router = APIRouter()

# 使用路由器来注册自己的路由
@product_router.get("/product")
def test_get():
    print("处理对/product的GET请求")
    return {"message": "product返回的数据。。。"}
# 后面还可以定义更多的路由