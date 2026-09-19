from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf

@dataclass
class ColumnConfig:
    name: str
    role: str
    description: str
    alias: list[str]
    sync: bool

@dataclass
class TableConfig:
    name: str
    role: str
    description: str
    columns: list[ColumnConfig]

@dataclass
class MetricConfig:
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]

@dataclass
class MetaConfig:
    tables: list[TableConfig]
    metrics: list[MetricConfig]

_yaml_path = Path(__file__).parents[2] / 'conf' / 'meta_config.yaml'

_yaml_data = OmegaConf.load(_yaml_path)

meta_config: MetaConfig = OmegaConf.to_object(OmegaConf.merge(MetaConfig, _yaml_data))

if __name__ == '__main__':
    print(meta_config)
    print(meta_config.metrics[0].description)















