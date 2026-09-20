from langchain.chat_models import init_chat_model

from app.conf.app_config import app_config

# ==========================================================================
# 大模型实例
# --------------------------------------------------------------------------
# 原实现是 `init_chat_model(model=..., api_key=...)`，它按**模型名**推断厂商。
# 微调后的模型名是自定义的（如 `qwen3-8b-lora`），推断不出来，
# 于是不会带上 base_url，会静默去连默认地址 —— 表现为"请求发不出去/超时"，
# 而不是一条明确的报错。所以这里**显式按 provider 选类**。
#
# 支持的两种 provider：
#   * openai_compatible —— 任何 OpenAI 兼容端点：
#       微调后的 Qwen3-8B（finetune/serve_qwen.py 或 vLLM 暴露的 /v1）
#       DeepSeek 官方云端（base_url 填 https://api.deepseek.com/v1）
#   * deepseek —— 归一到 openai_compatible（DeepSeek 就是 OpenAI 兼容协议），
#       保留这个分支只是为了兼容旧配置里写 provider: deepseek 的情况。
#
# 关键点：本地微调端点的 api_key 是**占位符**，不能因为缺 key 就启动失败。
# ==========================================================================

_provider = (app_config.llm.provider or "openai_compatible").strip().lower()
_base_url = (app_config.llm.base_url or "").strip()
_api_key = (app_config.llm.api_key or "").strip()

# 没配 base_url 时按 provider 兜底
if not _base_url:
    _base_url = ("https://api.deepseek.com/v1" if _provider == "deepseek"
                 else "http://127.0.0.1:8000/v1")

# 本地端点不校验 key，给个占位；DeepSeek 云端缺 key 会在调用时 401，
# 比"启动即崩"更容易定位
if not _api_key:
    _api_key = "EMPTY"

llm = init_chat_model(
    model=app_config.llm.model_name,
    model_provider="openai",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0,
)


def describe() -> str:
    """一行摘要，用于启动日志确认"到底连到了哪里"。"""
    return f"provider={_provider} model={app_config.llm.model_name} base_url={_base_url}"


if __name__ == '__main__':
    print(describe())
    result = llm.invoke('你是谁?')
    print(result.content)
