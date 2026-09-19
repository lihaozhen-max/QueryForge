from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import app_config
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
"""
用来操作qdrant数据库中指标信息数据的持久层模块
"""
class MetricQdrantRepository:
    collection_name = "data-agent_metric_collection2"
    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def create_collection(self):
        client = self.client
        collection_name = self.collection_name

        # 创建集合 如果存在先删除 方便反复测试

        if await client.collection_exists(collection_name=collection_name):
            await client.delete_collection(collection_name=collection_name)
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=app_config.qdrant.embedding_size,
                distance=models.Distance.COSINE
            ),
        )


    # 批量插入多个向量及其数据
    async def insert_metric_info_vectors(self, ids: list[str], payloads: list[MetricInfoQdrant],
                                         vectors: list[list[float]]):
        client = self.client
        collection_name = self.collection_name

        await self.create_collection()
        # 需要分批处理
        batch_size =10
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i: i + batch_size]
            batch_payloads = payloads[i: i+batch_size]
            batch_vectors = vectors[i:i+batch_size]

            # 批量插入多个向量
            await client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=batch_ids[j],
                        payload=batch_payloads[j],
                        vector=batch_vectors[j],
                    )
                    for j in range(len(batch_ids))
                ],
            )



    # 搜索

    async def search(self, keyword_vector: list[float]) -> list[MetricInfoQdrant]:
        # 搜索/查询
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=keyword_vector,
            score_threshold=0.6,  # 得分的最低值
        )

        # points = result.points
        # print(points)
        # for point in points:
        #     print(point.payload)
        return [MetricInfoQdrant(**item.payload) for item in result.points]




