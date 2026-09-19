from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.nodes.recall_column import recall_column
from app.agent.state import DataAgentState, TableInfoState, ColumnInfoState, MetricInfoState
from app.core.log import logger

from app.models.es.value_info_es import ValueInfoES
from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant

"""
1.收集一个最完整的字段信息列表 去重合并 dict[column_id,ColumnlnfoQdrant]
1.1.收集召回的字段信息列表
1.2.收集召回的指标信息列表联的字段信息列表
1.3.收集召回的字段值信息列表对应的字段信息列表=》将字段值保存到字段的值的样例列表中
1.4.收集相表的主键和外键字段信息列表
2.根据收集所有字段信息列表生成:带字段信息列表的表信息列表
2.1.对收集的字段信息列表进行按表id进行分组:dict[table_id, list[ColumnlnfoQdrant]]
2.2.生成带字段信息列表的表信息列表 -》table_infos:list[TablelnfoState]
    根据表id查询meta得到表信息
    根据当前字段信息列表·生成ColumnlnfoState类型的字段状态信息列表
"""





async def merge_retrieved_info(state: DataAgentState, runtime:Runtime[DataAgentContext]):
    # 做自定义输出
    runtime.stream_writer({"stage": "合并召回"})

    try:
        recall_columns: list[ColumnInfoQdrant] = state['recall_columns']
        recall_metrics: list[MetricInfoQdrant] = state['recall_metrics']

        recall_values: list[ValueInfoES] = state['recall_values']
        meta_mysql_repo = runtime.context['meta_mysql_repo']

        # 1. 收集一个最完整的字段信息列表 去重合并 dict[column_id,ColumnlnfoQdrant]
        # 1.1. 收集召回的字段信息列表
        column_infos_dict: dict[str, ColumnInfoQdrant] = {column_info['id']: column_info for column_info in recall_columns}

        # 1.2. 收集召回的指标信息列表联合的字段信息列表
        for metric in recall_metrics:
            for column_id in metric['relevant_columns']:
                if column_id not in column_infos_dict:
                    # 查询得到当前字段的字段信息
                    column_info_mysql: ColumnInfoMySQL = await meta_mysql_repo.get_column_info_by_column_id(column_id)
                    # 转换格式添加到字典列表中
                    column_infos_dict[column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)

        # 1.3. 收集召回的字段值信息列表对应的字段信息列表=》将字段值保存到字段的值的样例列表中
        for value_info in recall_values:
            column_id = value_info['column_id']
            if column_id not in column_infos_dict:
                column_info_mysql: ColumnInfoMySQL = await meta_mysql_repo.get_column_info_by_column_id(column_id)
                column_infos_dict[column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)
            value = value_info['value']
            examples = column_infos_dict[column_id]['examples']
            
            if value not in examples:
                examples.append(value)
                
        
        # 2.根据收集所有字段信息列表生成:带字段信息列表的表信息列表
        # 2.1.对收集的字段信息列表进行按表id进行分组:dict[table_id, list[ColumnlnfoQdrant]]
        table_column_infos_dict: dict[str, list[ColumnInfoQdrant]] = {}
        for column_info in column_infos_dict.values():
            table_id = column_info['table_id']
            if table_id not in table_column_infos_dict:
                table_column_infos_dict[table_id] = []
            table_column_infos_dict[table_id].append(column_info)
        # 1.4.收集相表的主键和外键字段信息列表
        # 2.2.生成带字段信息列表的表信息列表 -》table_infos:list[TablelnfoState]
        table_infos: list[TableInfoState] = []
        for table_id, column_infos in table_column_infos_dict.items():
            # 查询得到当前表的主键和外键字段信息列表
            key_column_infos: list[ColumnInfoQdrant] = await meta_mysql_repo.get_key_column_info(table_id)
            # 添加对应的字段列表中
            for key_column_info in key_column_infos:
                column_id = key_column_info.id
                if column_id not in column_infos_dict:
                    column_infos.append(_convert_column_info_mysql_to_qdrant(key_column_info))

            # 根据表id查询meta得到表信息
            table_info_mysql: TableInfoMySQL = await meta_mysql_repo.get_table_info(table_id)
            # 根据当前字段信息列表·生成ColumnlnfoState类型的字段状态信息列表
            columns: list[ColumnInfoState] = [
                _convert_column_info_qdrant_to_state(item)
                for item in column_infos
            ]

            # 创建对象添加到列表中
            table_infos.append(TableInfoState(
                name=table_info_mysql.name,
                role=table_info_mysql.role,
                description=table_info_mysql.description,
                columns=columns
            ))

        # 整理指标信息列表
        metric_infos: list[MetricInfoState] = [
            _convert_metric_info_qdrant_to_state(item)
            for item in recall_metrics
        ]

        logger.info(f"合并召回完成：{table_infos}")
        logger.info(f"合并召回完成：{metric_infos}")

        return {"table_infos": table_infos, "metric_infos": metric_infos}




        return {}
    except Exception as e:
        logger.error(f"合并召回失败： {str(e)}")
        raise


def _convert_metric_info_qdrant_to_state(recall_metric:MetricInfoQdrant):
    return MetricInfoState(
        name=recall_metric['name'],
        description=recall_metric["description"],
        relevant_columns=recall_metric["relevant_columns"],
        alias=recall_metric["alias"]
    )




def _convert_column_info_qdrant_to_state(column:ColumnInfoQdrant)->ColumnInfoState:
    return ColumnInfoState(
        name=column['name'],
        type=column['type'],
        role=column["role"],
        examples=column["examples"],
        description=column["description"],
        alias=column["alias"]
    )

def _convert_column_info_mysql_to_qdrant(column_info_mysql: ColumnInfoMySQL) -> ColumnInfoQdrant:
    return ColumnInfoQdrant(
        id = column_info_mysql.id,
        name = column_info_mysql.name,
        description=column_info_mysql.description,
        role=column_info_mysql.role,
        type=column_info_mysql.type,
        examples=column_info_mysql.examples,
        table_id=column_info_mysql.table_id,
        alias=column_info_mysql.alias
    )












