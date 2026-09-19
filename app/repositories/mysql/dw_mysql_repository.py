from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

"""
用来操作mysql数据库中dw库数据的持久层模块
"""

class DWMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    # 得到指定表中所有字段的类型
    async def get_column_types(self, table_name: str) -> dict[str, str]:
        sql = f'SHOW COLUMNS FROM `{table_name}`'
        result = await self.session.execute(text(sql))

        return {item.Field: item.Type for item in result.all()}

    # 得到指定表的指定字段的指定数量的字段值
    async def get_column_values(self, table_name: str, column_name: str, limit: int = 10) -> list:
        sql = f'SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT {limit}'
        result = await self.session.execute(text(sql))
        return result.scalars().all()


    # 获取数据库的版本和名称信息
    async def get_db_info(self)-> dict[str,str]:
        sql = 'select version()'
        result = await self.session.execute(text(sql))
        version: str = result.scalar()

        dialect = self.session.bind.dialect.name
        return {"version": version, "dialect": dialect}



    async def validate_sql(self,sql:str):
        """
        使用explain 关键字校验sql
        # 它利用数据库的解析器,能发现表不存在,字段名,JOIN语法错误等实际问题
        # 不会执行SQL
        :param sql:
        :return:
        """
        await self.session.execute(text(f'explain {sql}'))

    async def execute_sql(self, sql):
        """
        执行一条查询的SQL
        :param sql:
        :return:
        """
        result = await self.session.execute(text(sql))
        # return [dict(row) for row in result.mappings().all()]
        return  [dict(item) for item in result.mappings().all()]









