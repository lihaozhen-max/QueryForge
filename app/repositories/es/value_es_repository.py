from elasticsearch import AsyncElasticsearch
from app.models.es.value_info_es import ValueInfoES

"""
用来操作ES数据库中字段值数据的持久层模块
"""
class ValueESRepository:
    index_name = 'data-agent-value_index2'
    mappings = {
        "dynamic": False,
        "properties": {
            "id": {"type": "keyword", "index": False},
            "value": {"type": "text", "analyzer": "ik_max_word", "index": True},
            "type": {"type": "keyword", "index": False},
            "column_id": {"type": "keyword", "index": False},
            "column_name": {"type": "keyword", "index": False},
            "table_id": {"type": "keyword", "index": False},
            "table_name": {"type": "keyword", "index": False},
        }
    }

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    async def create_index(self):
        client = self.client
        index_name = self.index_name

        if await client.indices.exists(index=index_name):
            await client.indices.delete(index=index_name)
        await client.indices.create(
            index=index_name,
            mappings=self.mappings
        )

    # 批量插入多个字段值信息列表
    async def insert_value_infos(self, value_infos: list[ValueInfoES]):
        client = self.client
        index_name = self.index_name

        # 确保创建索引库
        await self.create_index()

        # 批次的数量
        batch_size = 30
        for i in range(0, len(value_infos), batch_size):
            batch_value_infos = value_infos[i:i + batch_size]

            # 遍历value_infos, 收集一个operations
            operations: list = []
            for value_info in batch_value_infos:
                operations.append({
                    'index':{
                        "_index": index_name,
                        # 显式指定文档id, 使重复构建时同一字段值覆盖同一文档而不是不断产生重复文档
                        "_id": value_info["id"]
                    }
                })
                operations.append(value_info)

            # 批量插入多个文档
            await client.bulk(
                operations=operations,
            )



    # 搜索
    async def search(self, keyword)->list[ValueInfoES]:
        result = await self.client.search(
            index=self.index_name,
            query={
                "match": {
                    "value": keyword
                }
            },
        )
        # print(result)
        # print(result["hits"]["hits"][0]["_source"])

        return [ValueInfoES(**item["_source"]) for item in result["hits"]["hits"]]

















