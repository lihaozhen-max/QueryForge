from sqlalchemy import Select

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL

"""
用来操作mysql数据库中meta库数据的持久层模块
"""
class MetaMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    def save_table_infos(self, table_infos: list[TableInfoMySQL]):
        self.session.add_all(table_infos)

    def save_column_infos(self, column_infos: list[ColumnInfoMySQL]):
        self.session.add_all(column_infos)

    def save_metric_infos(self, metric_infos: list[MetricInfoMySQL]):
        self.session.add_all(metric_infos)

    def save_column_metric(self, column_metric: list[ColumnMetricMySQL]):
        self.session.add_all(column_metric)



    async def get_column_info_by_column_id(self, column_id: str) -> ColumnInfoMySQL:
        return await self.session.get(ColumnInfoMySQL, column_id)

    # 获取指定表中的主键和外键字段信息列表
    async def get_key_column_info(self, table_id):
        result = await self.session.execute(
            Select(ColumnInfoMySQL)
            .where(ColumnInfoMySQL.table_id == table_id)
            .where(ColumnInfoMySQL.role.in_(['primary_key', 'foreign_key']))
        )

        return result.scalars().all()

    async def get_table_info(self, table_id: str)-> TableInfoMySQL:
        return await self.session.get(TableInfoMySQL, table_id)










    # AI写的(老师没写)
    async def clear_all(self):
        # 重复构建前清空旧数据，避免主键冲突；先删关联表再删主表
        await self.session.execute(delete(ColumnMetricMySQL))
        await self.session.execute(delete(MetricInfoMySQL))
        await self.session.execute(delete(ColumnInfoMySQL))
        await self.session.execute(delete(TableInfoMySQL))









