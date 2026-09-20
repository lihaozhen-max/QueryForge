from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


# ==================== 日志配置模型 ====================
@dataclass
class File:
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    enable: bool
    level: str

@dataclass
class LoggingConfig:
    file: File
    console: Console


# ==================== database配置模型 ====================

@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

# ==================== Qdrant 配置模型 ====================

@dataclass
class QdrantConfig:
    host: str
    port: int
    embedding_size: int


# ==================== Embedding 配置模型 ====================

@dataclass
class EmbeddingConfig:
    host: str
    port: int
    model: str


# ==================== ES 配置模型 ====================

@dataclass
class ESConfig:
    host: str
    port: int
    index_name: str


# ==================== LLM 配置模型 ====================

@dataclass
class LLMConfig:
    """
    大模型配置。

    `provider` 决定走哪条链路：
      * `openai_compatible`（默认）：任意 OpenAI 兼容端点。
        微调后的 Qwen3-8B 由 vLLM 暴露的 `/v1` 接口就属于这类，
        **不需要任何 API Key**；本机联调用的假服务也是这类。
      * `deepseek`：DeepSeek 官方云端接口。

    为什么不用 langchain 的 `init_chat_model(model=...)` 自动推断：
    它按**模型名**猜厂商，而微调模型的 model 名是自定义的（如 `qwen3-8b-lora`），
    猜不出来就不会带上 base_url。这里显式选类，避免"连到了错的地址"这类静默错误。
    """

    model_name: str
    api_key: str
    provider: str = "openai_compatible"
    # 空字符串 = 用该 provider 的默认地址；本地 vLLM 通常要显式给
    # 例如 http://127.0.0.1:8000/v1
    base_url: str = ""


# ==================== 应用总配置模型 ====================

@dataclass
class AppConfig:
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig

_yaml_path = Path(__file__).parents[2] / 'conf' / 'app_config.yaml'

_yaml_data = OmegaConf.load(_yaml_path)

app_config: AppConfig = OmegaConf.to_object(OmegaConf.merge(AppConfig, _yaml_data))

if __name__ == '__main__':
    print(app_config, type(app_config))
    print(app_config.logging.file.level)























