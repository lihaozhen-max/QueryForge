import asyncio
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

"""
langgraph相关重要概念
    状态图：StateGraph的实例
    节点：实现特定功能的函数
    状态：包含多个可变数据的对象，在图的各个节点上传递
    边： 它是A->B2个节点的连线, A执行后执行B，且将最新的状态数据传递给B
节点函数的参数：
    state: 状态对象   自定义类型
    runtime: Runtime  运行时对象，包含:
        stream_writer: 向调用者输出自定义数据的函数
        store:实现跨会话长期记忆的对象，默认是保存在内存中
        context: 包含固定数据或依赖的外部模块的对象   自定义类型
    config：ConfigRunnable 保存配置的对象，比如执行时指定的thread_id
stream_mode:
    updates: 外部得到是节点返回的更新
    values: 外部得到是节点返回后合并后的state值
    custom: 外部得到是stream_writer指定的自定义数据
"""

# 定义状态类型
# 包含一些可变的数据
class Mystate(TypedDict):
    query: str
    keywords: list[str]
    sql: str
    error: str


# 自定义context类型
# 包含一些不可变的数据或者依赖模块 => 节点需要使用
class MyContext(TypedDict):
    db_name: str


# 定义节点函数
def extract_keywords(state: Mystate, runtime: Runtime[MyContext], config: RunnableConfig):
    print(runtime)
    print(config)
    print(f'db_name={runtime.context['db_name']}')

    # 自定义输出

    runtime.stream_writer('提取关键字')

    query = state['query']
    # 进行关键字的提取
    print(f'对[{query}]提取关键字')
    keywords: list[str] = ['我', '是', '谁']

    return {'keywords': keywords}

def generate_sql(state: Mystate, runtime: Runtime):
    runtime.stream_writer('生成SQL')

    sql = 'select * from xxx'
    print(f'keywords:{state['keywords']}')
    print(f'生成SQL: {sql}')

    return {'sql': sql}




state_graph = StateGraph(
    state_schema=Mystate,
    context_schema=MyContext
)


# 添加节点
state_graph.add_node("extract_keywords", extract_keywords)
state_graph.add_node("generate_sql", generate_sql)

# 添加边
state_graph.add_edge(START, "extract_keywords")
state_graph.add_edge("extract_keywords", "generate_sql")
state_graph.add_edge("generate_sql", END)

# 编译状态图
compiled_graph = state_graph.compile()

# 执行
if __name__ == '__main__':
    async def test():
        # 创建state对象
        state = Mystate(query="我是谁？")
        # 创建context对象
        context = MyContext(db_name="dw")

        async for chunk in compiled_graph.astream(
            input=state,
            context=context,
            config={"thread_id": "abc"},  # 推荐会话id的配置
            stream_mode="custom"   # 自定义节点输出
        ):
            print("----", chunk)


    asyncio.run(test())













