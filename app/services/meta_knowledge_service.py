import asyncio
import hashlib
import uuid

import aiohttp
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from app.clients.embedding_client_manager import embedding_client_manager
from app.conf.meta_config import meta_config, TableConfig, MetricConfig
from app.core.log import logger
from app.models.es.value_info_es import ValueInfoES
from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class MetaKnowledgeService:
    def __init__(
            self,
            dw_mysql_repo: DWMysqlRepository,
            meta_mysql_repo: MetaMysqlRepository,
            value_es_repo: ValueESRepository,
            column_qdrant_repo: ColumnQdrantRepository,
            metric_qdrant_repo: MetricQdrantRepository,

            embedding_client: HuggingFaceEndpointEmbeddings
    ):
        self.dw_mysql_repo = dw_mysql_repo
        self.meta_mysql_repo = meta_mysql_repo
        self.value_es_repo = value_es_repo
        self.column_qdrant_repo = column_qdrant_repo
        self.metric_qdrant_repo = metric_qdrant_repo
        self.embedding_client = embedding_client


    # 调用embedding服务生成向量, 失败时自动重建客户端并指数退避重试
    async def _embed_with_retry(self, batch_texts: list[str], max_retries: int = 3) -> list[list[float]]:
        for attempt in range(max_retries):
            try:
                return await self.embedding_client.aembed_documents(batch_texts)
            except (aiohttp.ClientError, ConnectionError, OSError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(f'embedding请求失败且重试{max_retries}次仍失败, 放弃: {e}')
                    raise
                # 服务端可能已崩溃重启(如TEI队列打满panic), 旧连接的session可能已失效, 需要重建客户端
                logger.warning(f'embedding请求失败(第{attempt + 1}次): {e}, 重建embedding客户端, {2 ** attempt}秒后重试...')
                embedding_client_manager.init_client()
                self.embedding_client = embedding_client_manager.client
                await asyncio.sleep(2 ** attempt)


    # 构建元数据知识库的业务方法
    """
    1. 处理表和字段相关信息数据
        1.1 将表信息和字段信息保存到meta库
        1.2 将字段信息保存到qdrant建立向量索引
        1.3 讲字段值保存到es建立全文索引
    2. 处理指标相关信息数据
        2.1 将指标信息保存到meta库
        2.2 将指标信息保存到qdrant建立向量索引
        
    """

    async def build(self):
        # 先清空meta库旧数据，保证重复构建不会主键冲突
        await self.meta_mysql_repo.clear_all()

        column_infos: list[ColumnInfoMySQL] = await self._save_table_infos_to_meta(meta_config.tables)
        logger.info('将表信息和字段信息保存meta库成功')

        await self._save_column_infos_to_qdrant(column_infos)
        logger.info('将字段信息保存到qdrant建立向量索引成功')

        await self._save_value_infos_to_es(column_infos,meta_config.tables)
        logger.info('将字段值保存到es建立全文索引成功')

        metric_infos: list[MetricInfoMySQL] = self._save_metric_infos_to_meta(meta_config.metrics)
        logger.info('将指标信息保存到meta库成功')

        await self._save_metric_infos_to_qdrant(metric_infos)
        logger.info('将指标信息保存到qdrant建立向量索引成功')














    async def _save_table_infos_to_meta(self,tables:list[TableConfig]) ->list[ColumnInfoMySQL]:
        # 遍历tables准备表信息列表和字段信息列表
        table_infos: list[TableInfoMySQL] = []
        column_infos: list[ColumnInfoMySQL] = []

        for table in tables:
            table_infos.append(TableInfoMySQL(
                id=table.name,
                name=table.name,
                role=table.role,
                description=table.description
            ))
            # 查询当前表中所有字段的类型 dict[字段名:字段类型]
            column_types: dict[str,str] = await self.dw_mysql_repo.get_column_types(table.name)

            for column in table.columns:
                # 查询当前字段的前10个字段值
                examples:list = await self.dw_mysql_repo.get_column_values(table.name, column.name)

                column_infos.append(ColumnInfoMySQL(
                    id=f'{table.name}.{column.name}',
                    name=column.name,
                    type=column_types[column.name],
                    role=column.role,
                    examples=examples,
                    description=column.description,
                    alias=column.alias,
                    table_id=table.name

                ))
        self.meta_mysql_repo.save_table_infos(table_infos)
        self.meta_mysql_repo.save_column_infos(column_infos)

        return column_infos





    async def _save_column_infos_to_qdrant(self, column_infos:list[ColumnInfoMySQL]):
        # 遍历column_infos准备ids / payloads/ vectors
        # list[{id, payload, text}]

        temp_list: list[dict] = []
        for column_info in column_infos:

            payload = ColumnInfoQdrant(
                id=column_info.id,
                name=column_info.name,
                type=column_info.type,
                role=column_info.role,
                examples=column_info.examples,
                description=column_info.description,
                alias=column_info.alias,
                table_id=column_info.table_id
            )

            temp_list.append({
                'id': uuid.uuid4(),
                'payload': payload,
                'text': column_info.name
            })

            temp_list.append({
                'id': uuid.uuid4(),
                'payload': payload,
                'text': column_info.description
            })

            for alia in column_info.alias:
                temp_list.append({
                    'id': uuid.uuid4(),
                    'payload': payload,
                    'text': alia
                })

        # 取出手机的各个数据列表
        ids: list[str] = [item['id'] for item in temp_list]
        payloads: list[ColumnInfoQdrant] = [item['payload'] for item in temp_list]
        texts: list[str] = [item['text'] for item in temp_list]

        # 生成向量列表(需要进行分批处理, batch_size不宜过大, 避免把TEI的请求队列打满导致服务崩溃)
        vectors: list[list[float]] = []
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors: list[list[float]] = await self._embed_with_retry(batch_texts)
            vectors.extend(batch_vectors)


        await self.column_qdrant_repo.insert_column_info_vectors(ids, payloads, vectors)




























    async def _save_value_infos_to_es(self, column_infos:list[ColumnInfoMySQL], tables:list[TableConfig]):
        # 收集所有字段是否需要进行es索引的标识
        column_sync_dict:dict[str,bool] = {}
        for table in tables:
            for column in table.columns:
                column_sync_dict[f'{table.name}.{column.name}'] = column.sync
        # 遍历column_infos, 收集value_infos列表
        value_infos: list[ValueInfoES] = []
        for column_info in column_infos:
            if column_sync_dict[column_info.id]:
                # 查询得到当前字段的所有值
                values = await self.dw_mysql_repo.get_column_values(column_info.table_id, column_info.name,10000)
                # 为每个值创建ValueInfoES对象, 并添加到列表中
                for value in values:
                    value_infos.append(ValueInfoES(
                        # 由字段id + 字段值生成稳定且唯一的文档id: 同一字段值重复构建时id不变, 避免重复文档
                        id=_build_value_doc_id(column_info.id, value),
                        value = value,
                        type = column_info.type,
                        column_id = column_info.id,
                        column_name = column_info.name,
                        table_id=column_info.table_id,
                        table_name=column_info.table_id
                    ))

        await self.value_es_repo.insert_value_infos(value_infos)











    def _save_metric_infos_to_meta(self, metrics: list[MetricConfig]) -> list[MetricInfoMySQL]:

    # 遍历metrics收集指标信息列表和字段指标信息列表
        metric_infos: list[MetricInfoMySQL] = []
        column_metrics: list[ColumnMetricMySQL] = []
        for metric in metrics:
            metric_infos.append(MetricInfoMySQL(
                id=metric.name,
                name=metric.name,
                description=metric.description,
                relevant_columns=metric.relevant_columns,
                alias=metric.alias
            ))
            for column_id in metric.relevant_columns:
                column_metrics.append(ColumnMetricMySQL(
                    column_id=column_id,
                    metric_id=metric.name
                ))

        self.meta_mysql_repo.save_metric_infos(metric_infos)
        self.meta_mysql_repo.save_column_metric(column_metrics)

        return metric_infos



    async def _save_metric_infos_to_qdrant(self, metric_infos:list[MetricInfoMySQL]):
        # 遍历metric_infos准备ids / payloads / vectors
        # list[{id,payload,text}]
        temp_list: list[dict] = []
        for metric_info in metric_infos:
            payload = MetricInfoQdrant(
                id=metric_info.id,
                name=metric_info.name,
                description=metric_info.description,
                alias=metric_info.alias,
                relevant_columns=metric_info.relevant_columns
            )
            #name
            temp_list.append({
                'id': uuid.uuid4(),
                'payload': payload,
                'text': metric_info.name
            })

            # description
            temp_list.append({
                'id': uuid.uuid4(),
                'payload': payload,
                'text': metric_info.description
            })

            for alia in metric_info.alias:
                temp_list.append({
                    "id": uuid.uuid4(),
                    "payload": payload,
                    "text": alia
                })

        # 取出各个数据列表
        ids: list[str] = [item['id'] for item in temp_list]
        payloads: list[MetricInfoQdrant] = [item['payload'] for item in temp_list]
        texts: list[str] = [item['text'] for item in temp_list]

        # 生成向量列表 (需要进行分批处理, batch_size不宜过大)
        vectors: list[list[float]] = []
        batch_size = 4

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors: list[list[float]] = await self._embed_with_retry(batch_texts)
            vectors.extend(batch_vectors)

        # 保存到qdrant
        await self.metric_qdrant_repo.insert_metric_info_vectors(ids, payloads, vectors)


def _build_value_doc_id(column_id: str, value) -> str:
    """由字段id和字段值构造稳定的ES文档id(32位十六进制, 满足ES _id长度限制)"""
    raw = f'{column_id}\u0000{value}'
    return hashlib.md5(raw.encode('utf-8')).hexdigest()










