# 负责定义查询接口请求体结构
from pydantic import BaseModel

# 封装请求体函数
class QuerySchema(BaseModel):
    query: str
