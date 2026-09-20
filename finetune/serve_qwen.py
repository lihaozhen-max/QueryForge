#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 transformers 把「基座 + LoRA」包装成 **OpenAI 兼容**的 HTTP 服务。

为什么不用 vLLM
---------------
2026-09-20 在租卡机器上实测：`pip install vllm` 会把环境整体升一级
（torch → 2.13+cu130、transformers → 5.17、tokenizers → 0.23.2），
而 torchaudio 没跟着升，导致 transformers 一 import 就崩：

    RuntimeError: PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8

补上配套的 torchaudio 之后 vLLM 自身的 import 又报 `duplicate template name`。
结论：**这套镜像上 vLLM 不划算**，而 transformers 这条链路
（基座 + LoRA + bf16）已经用 `predict_with_model.py` 验证过能跑，
所以直接用它包装成服务，**零新增依赖**。

代价是没有 vLLM 的高并发与连续批处理。对"自己跑通一次流程"足够。

暴露的接口（够 langchain 用）
----------------------------
    GET  /v1/models
    POST /v1/chat/completions      ← 支持 stream 和非 stream

用法（租卡机器）
----------------
    python serve_qwen.py
    python serve_qwen.py --adapter /root/autodl-tmp/output/qwen3-8b-lora-v2-1ep-ep1
    python serve_qwen.py --port 8000 --served-name qwen3-8b-lora

启动后：
    curl -s http://127.0.0.1:8000/v1/models

关于思维链（重要）
------------------
Qwen3 的 chat_template 里 `enable_thinking` 语义是反直觉的：
  * `False` → 渲染成 `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
             即**预先塞一个空思维链块**，命令模型"别想，直接答"
  * `True`  → 渲染成 `<|im_start|>assistant\\n`，让模型自己决定要不要想

本项目默认 `--thinking`（等价 True）。理由：实测开着思维链时，
模型会走维表（`IN (SELECT date_id FROM dim_date WHERE month = 2)`），
关掉则退化成硬编码日期（`BETWEEN 20250201 AND 20250228`）。

输出处理：思维链会被剥掉，只把最终答案作为 `content` 返回。
两种结束标记都处理（ASCII `</think>` 与 Qwen 专用的 `｜end▁of▁thinking｜`）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ==========================================================================
# 注意：fastapi 的导入**必须在模块顶层**，不能放进函数里。
# --------------------------------------------------------------------------
# 本文件顶部有 `from __future__ import annotations`，它把所有注解变成**字符串**，
# fastapi 在运行时用 get_type_hints 去**模块全局命名空间**解析这些字符串。
# 如果把 `from fastapi import Request` 写在 build_app() 内部，`Request` 只存在于
# 函数局部，全局里找不到 —— fastapi 无法识别 `request: Request` 是请求对象，
# 就退化成把它当成一个**必填的查询参数**，于是 POST 报 422：
#     {"detail":[{"type":"missing","loc":["query","request"],"msg":"Field required"}]}
# 这个坑很隐蔽：代码看着完全正常，报错却指向"缺一个叫 request 的查询参数"。
# 之前踩过（2026-09-20），所以这里显式留在顶层。
# ==========================================================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ==========================================================================
# 输出清洗：把思维链剥掉，只留最终答案
# ==========================================================================
_THINK_ASCII = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
# Qwen 专用结束标记：｜ 是全角竖线，▁ 是 U+2581
_THINK_QWEN = re.compile(r"<think\b[^>]*>.*?<｜end▁of▁thinking｜>", re.DOTALL | re.IGNORECASE)
_THINK_LEFTOVER = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_FENCE = re.compile(r"^```(?:sql|json)?\s*|\s*```$", re.IGNORECASE)


def strip_thinking(text: str, thinking: bool) -> str:
    """剥掉思维链与 markdown 围栏。thinking=False 时不做剥离（提示词里已带空块）。"""
    t = (text or "").strip()
    if thinking:
        t = _THINK_ASCII.sub("", t)
        t = _THINK_QWEN.sub("", t)
        if "<think" in t.lower():
            # 不成对时：从开头丢到第一个结束标记
            t = re.sub(r"^.*?(?:</think\s*>|<｜end▁of▁thinking｜>)", "", t, flags=re.DOTALL | re.IGNORECASE)
        t = _THINK_LEFTOVER.sub("", t)
        t = t.replace("<｜end▁of▁thinking｜>", "")
    return _FENCE.sub("", t).strip()


# ==========================================================================
# 模型
# ==========================================================================
class Engine:
    def __init__(self, model_dir: str, adapter: str | None, thinking: bool,
                 max_len: int, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.thinking = thinking
        self.max_len = max_len
        # GPU 上逐个请求串行跑，避免并发把显存打爆
        self.lock = threading.Lock()

        t0 = time.time()
        print(f"[engine] 加载 tokenizer：{model_dir}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        print(f"[engine] 加载模型（{dtype}）……", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=getattr(torch, dtype),
            device_map="cuda",
            trust_remote_code=True,
        )
        if adapter:
            from peft import PeftModel
            print(f"[engine] 挂载 LoRA：{adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter)
            # 合并进权重，省一次每步的 LoRA 计算
            self.model = self.model.merge_and_unload()
        self.model.eval()
        print(f"[engine] 就绪，用时 {time.time()-t0:.0f}s，"
              f"显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    def chat(self, messages: list[dict], max_new_tokens: int,
             temperature: float, stop: list[str] | None) -> tuple[str, int, int]:
        """返回 (文本, prompt_token 数, completion_token 数)"""
        torch = self.torch
        kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tok.apply_chat_template(
                messages, enable_thinking=self.thinking, **kwargs)
        except TypeError:
            # 模板不认这个参数（非 Qwen3）时退回默认
            text = self.tok.apply_chat_template(messages, **kwargs)

        with self.lock:
            ids = self.tok(text, return_tensors="pt").to(self.model.device)
            n_prompt = int(ids["input_ids"].shape[1])
            if n_prompt > self.max_len:
                raise ValueError(
                    f"提示词 {n_prompt} token 超过 max_len={self.max_len}，请调大 --max-len")
            gen_kwargs: dict[str, Any] = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                repetition_penalty=1.05,
                pad_token_id=self.tok.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            with torch.no_grad():
                out = self.model.generate(**ids, **gen_kwargs)

        new_ids = out[0][n_prompt:]
        n_out = int(new_ids.shape[0])
        raw = self.tok.decode(new_ids, skip_special_tokens=True)
        content = strip_thinking(raw, self.thinking)

        if stop:
            for s in stop:
                if s and s in content:
                    content = content.split(s)[0]
        return content.strip(), n_prompt, n_out


# 模块级状态。SERVED_NAME 在 main() 里按命令行设置，
# 但 /v1/models 与响应体都要用它，所以必须是模块全局。
SERVED_NAME = "qwen3-8b-lora"
ENGINE: Engine | None = None


# ==========================================================================
# HTTP（FastAPI，项目本来就用它，零新增依赖）
# ==========================================================================
def build_app(default_max_tokens: int):
    app = FastAPI(title="Qwen3-8B LoRA (OpenAI compatible)", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": ENGINE is not None}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": SERVED_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }],
        }

    def _oa(content: str, n_prompt: int, n_out: int) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "total_tokens": n_prompt + n_out,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            return JSONResponse(
                {"error": {"message": "messages 不能为空", "type": "invalid_request_error"}},
                status_code=400)

        max_tokens = int(body.get("max_tokens") or default_max_tokens)
        temperature = float(body.get("temperature") or 0.0)
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        stream = bool(body.get("stream"))

        if ENGINE is None:
            return JSONResponse(
                {"error": {"message": "模型尚未加载完成", "type": "server_error"}},
                status_code=503)

        try:
            content, n_prompt, n_out = ENGINE.chat(messages, max_tokens, temperature, stop)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}},
                status_code=500)

        if not stream:
            return JSONResponse(_oa(content, n_prompt, n_out))

        # ---- 流式：langchain 会走这条 ----
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def gen():
            first = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            # 一次性把整段吐出来（本服务不做逐 token 流式）
            if content:
                chunk = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": SERVED_NAME,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            last = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                          "total_tokens": n_prompt + n_out},
            }
            yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> int:
    global ENGINE, SERVED_NAME

    ap = argparse.ArgumentParser(description="把基座+LoRA 包成 OpenAI 兼容服务")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 适配层目录；不给就是原版模型")
    ap.add_argument("--served-name", default="qwen3-8b-lora",
                    help="客户端 model 字段要填的名字")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-len", type=int, default=4096,
                    help="提示词最大 token 数，与训练 cutoff_len 对齐")
    ap.add_argument("--max-tokens", type=int, default=512, help="默认最大生成 token")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", default=True,
                    help="关闭思维链。注意这会让输入多一个空 <think> 块，"
                         "实测 SQL 质量会下降，除非确认训练时也这么喂")
    args = ap.parse_args()

    SERVED_NAME = args.served_name
    ENGINE = Engine(args.model, args.adapter, args.thinking, args.max_len, args.dtype)

    import uvicorn
    print(f"[http] 监听 http://{args.host}:{args.port}   model={SERVED_NAME}", flush=True)
    uvicorn.run(build_app(args.max_tokens), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
