import asyncio
import random
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import QdrantConfig, app_config

"""
用来操作qdrant向量库的客户端管理器模块
"""

class QdrantClientManager:
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None

    def init_client(self):
        self.client = AsyncQdrantClient(f"http://{self.config.host}:{self.config.port}")

    async def close(self):
        await self.client.close()

qdrant_client_manager = QdrantClientManager(app_config.qdrant)

if __name__ == '__main__':
    async def test():
        qdrant_client_manager.init_client()

        client = qdrant_client_manager.client

        collection_name = 'my_collection'

        if await client.collection_exists(collection_name=collection_name):
            await client.delete_collection(collection_name=collection_name)
        await  client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        )

        # 批量插入多个向量
        await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=i,
                    payload={
                        "color": "red" if i % 2 == 0 else "blue",
                    },
                    vector=[random.random() for _ in range(1024)],  # list[float]
                )
                for i in range(10)
            ],
        )

        # 搜索/查询
        result = await client.query_points(
            collection_name=collection_name,
            query=[random.random() for _ in range(1024)],
            limit=5,
            score_threshold=0.5,
            query_filter=models.Filter(
                must=[models.FieldCondition(key='color', match=models.MatchValue(value="red"))]
            ),
        )

        points = result.points
        print(points)
        for point in points:
            print(point.payload)

        await qdrant_client_manager.close()

    asyncio.run(test())















