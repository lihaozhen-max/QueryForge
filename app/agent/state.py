from typing import TypedDict

from app.models.es.value_info_es import ValueInfoES
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant

# 列信息封装实体
class ColumnInfoState(TypedDict):
    name: str
    type: str
    role: str
    examples: list
    description: str
    alias: list[str]


# 表信息封装实体
class TableInfoState(TypedDict):
    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]

# 指标信息封装实体
class MetricInfoState(TypedDict):
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]


# 当前日期时间的状态信息
class DateInfoState(TypedDict):
    date: str # 年月日
    weekday: str # 星期
    quarter: str # 季度  Q1-Q4

# 数据库信息
class DBInfoState(TypedDict):
    version: str # 版本号
    dialect: str # 数据库名称




class DataAgentState(TypedDict):
    query: str
    keywords: list[str]
    sql: str
    error: str
    correct_sql_count: int # 已执行的校正次数, 用于给"校验-校正"环设置上限
    recall_columns: list[ColumnInfoQdrant] # 召回的字段信息列表
    recall_metrics: list[MetricInfoQdrant] # 召回的指标信息列表
    recall_values: list[ValueInfoES] # 召回的字段值信息列表
    table_infos: list[TableInfoState] # 带字段信息列表的表信息列表
    metric_infos: list[MetricInfoState]
    date_info: DateInfoState # 当前日期时间信息
    db_info: DBInfoState # 数据库信息