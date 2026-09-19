import asyncio
from typing import Optional

from sqlalchemy import text, Select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

from app.conf.app_config import DBConfig, app_config
from app.models.mysql.table_info_mysql import TableInfoMySQL

"""用于操作mysql数据库的客户端管理器模块"""
class MysqlClientManager:
    def __init__(self, config: DBConfig):
        self.config = config
        self. client: Optional[AsyncEngine] = None
        self.session_factory:Optional[async_sessionmaker] = None

    def _get_url(self) -> URL:
        # 注意：原实现用手拼字符串且漏掉了 port，导致配置里的端口不生效、
        # 永远按驱动默认的 3306 连接（本机 3306 被其它 MySQL 占用时会误连并报鉴权失败）。
        # 这里改用 URL.create 显式带上端口，并由它负责用户名/密码的转义
        # （密码含 . @ : 等字符时手拼会出错）。
        return URL.create(
            drivername="mysql+asyncmy",
            username=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            query={"charset": "utf8mb4"},
        )

    def init_client(self):
        self.client = create_async_engine(
            self._get_url(),
            pool_size=10,
            max_overflow=5
        )
        self.session_factory = async_sessionmaker(
            self.client,
            autobegin=True,
            autoflush=False
        )

    async def close(self):
        await self.client.dispose()


dw_mysql_client_manager = MysqlClientManager(app_config.db_dw)
meta_mysql_client_manager = MysqlClientManager(app_config.db_meta)

if __name__ == '__main__':

    # 测试ORM 插入和查询

    async def test_orm():
        meta_mysql_client_manager.init_client()
        async with meta_mysql_client_manager.session_factory() as session:
            session: AsyncSession

            # 先清空表，避免重复插入主键报错
            await session.execute(text("DELETE FROM table_info"))
            await session.commit()

            table_info1 = TableInfoMySQL(
                id="dim_customer1",
                name="dim_customer1",
                role="dim",
                description="客户维度表1"
            )
            session.add(table_info1)
            # 插入多条数据
            table_info2 = TableInfoMySQL(
                id="dim_customer2",
                name="dim_customer2",
                role="dim",
                description="客户维度表2"
            )
            table_info3 = TableInfoMySQL(
                id="dim_customer3",
                name="dim_customer3",
                role="dim",
                description="客户维度表3"
            )

            session.add_all([table_info2, table_info3])

            await  session.commit()

            table_info = await session.get(TableInfoMySQL, "dim_customer1")

            print(table_info, table_info.description)

            result = await session.execute(
                Select(TableInfoMySQL)
                .limit(2)
            )

            table_infos: list[TableInfoMySQL] = result.scalars().all()
            print(table_infos, table_infos[0].description)

        await  meta_mysql_client_manager.close()


    async def test_orm2():
        meta_mysql_client_manager.init_client()

        async with meta_mysql_client_manager.session_factory() as session:
            session: AsyncSession

            table_info = await session.get (TableInfoMySQL,'dim_customer1')
            table_info.description = 'aaa'

            await session.delete(table_info)

            await session.commit()




        await meta_mysql_client_manager.close()

    async def test():
        dw_mysql_client_manager.init_client()
        assert dw_mysql_client_manager.session_factory
        async with dw_mysql_client_manager.session_factory() as session:
            sql = "select customer_name from dim_customer limit 2"
            result = await session.execute(text(sql))


            """
            result.all(): 返回[row, row], row是包含当前行的字段值的可遍历的对象
            result.mappings().all(): 返回[rowMapping, rowMapping], rowMapping是包含当前行的字段名和字段值的可遍历的对象
            result.scalars().all(): 返回[val, val], val是第一个字段值
            """

            rows = result.scalars().all()
            for val in rows:
                print(val)

        await dw_mysql_client_manager.close()




    asyncio.run(test())


















    # # 测试ORM 更新和删除
    #
    # async def test_orm2():
































