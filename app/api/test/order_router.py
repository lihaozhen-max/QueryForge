from fastapi import APIRouter
# 管理订单的多个路由的路由器
order_router = APIRouter()

# 使用路由器来注册自己的路由
@order_router.get('/order')
def get_order():
    print('处理对/order的GET请求')
    return {'message': 'order返回的数据...'}

# 后面可以定义更多路由