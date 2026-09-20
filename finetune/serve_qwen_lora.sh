#!/bin/bash
# ============================================================================
# 掌柜问数 · 用 vLLM 部署微调后的 Qwen3-8B（LoRA 适配层）
#
# 用法（租卡机器，任意目录）：
#     bash /root/autodl-tmp/serve_qwen_lora.sh
#
# 它做三件事：
#   1. 检查 vLLM 是否可用（没有就装）
#   2. 检查基座模型与 LoRA 适配层都在
#   3. 后台启动 vLLM，暴露 **OpenAI 兼容**接口
#
# 起好之后接口地址：
#     http://127.0.0.1:8000/v1          ← 本机（在租卡机器上）
#     http://<AutoDL映射的6006端口>/v1   ← 从你自己的电脑访问时用这个
#     模型名（model 字段）：qwen3-8b-lora
#
# 验证：
#     curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
#
# 为什么用 vLLM 而不是 transformers 直接推理
# -------------------------------------------
# * 暴露的是 **OpenAI 兼容接口**，应用侧只要改 base_url + model 就能切过去，
#   不需要为微调模型改任何业务代码。
# * 支持 `--enable-lora` 直接挂适配层，**不用导出 16GB 合并模型**
#   （之前导出合并模型时踩过磁盘满：No space left on device）。
# * 并发与吞吐远好于 transformers 逐条推理。
#
# 注意
# ----
# * `--lora-modules` 里 `名字=路径` 中，**等号左边就是客户端要填的 model 名**。
#   本项目约定用 `qwen3-8b-lora`，与 conf/app_config.yaml 的 qwen3.model_name 保持一致。
# * `--max-model-len 4096` 与训练时的 cutoff_len 对齐；实测最长提示词 2887 token，
#   加上生成 512 仍在 4096 内。
# ============================================================================
set -u

BASE=/root/autodl-tmp
MODEL_DIR="$BASE/models/Qwen3-8B"
ADAPTER_DIR="${ADAPTER_DIR:-$BASE/output/qwen3-8b-lora-v2-1ep-ep1}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b-lora}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"

mkdir -p "$BASE/logs"
LOG="$BASE/logs/vllm_${SERVED_NAME}.log"
PIDFILE="$BASE/logs/vllm_${SERVED_NAME}.pid"

say()  { echo "$*"; }
die()  { echo ""; echo "!!! 失败：$*"; echo "日志：$LOG"; exit 1; }

echo "=========================================================="
echo " 掌柜问数 · 部署微调 Qwen3-8B"
echo "=========================================================="
echo "  基座模型 : $MODEL_DIR"
echo "  LoRA 适配 : $ADAPTER_DIR"
echo "  模型名    : $SERVED_NAME   ← 客户端 model 字段填这个"
echo "  端口      : $PORT"
echo "  上下文长度 : $MAX_LEN"
echo ""

# ---- 1. 检查前置 ----
echo "[1/4] 前置检查"
[ -f "$MODEL_DIR/config.json" ] || die "缺少基座模型 $MODEL_DIR/config.json"
[ -f "$ADAPTER_DIR/adapter_config.json" ] || die "缺少适配层 $ADAPTER_DIR/adapter_config.json"
[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || die "缺少权重 $ADAPTER_DIR/adapter_model.safetensors"
echo "      基座与适配层就绪"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  die "已有 vLLM 在跑（pid $(cat "$PIDFILE")）。要重启请先：kill \$(cat $PIDFILE)"
fi
if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
  die "端口 $PORT 上已有服务在响应。换个 PORT，或先停掉它"
fi
echo "      端口 $PORT 空闲"
echo "      磁盘：$(df -h "$BASE" | tail -1)"

# ---- 2. 检查 vLLM ----
echo ""
echo "[2/4] 检查 vLLM"
if python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
  echo "      vLLM 已安装：$(python -c 'import vllm;print(vllm.__version__)')"
else
  echo "      未安装 vLLM，开始安装（用清华源，约 3-5 分钟）……"
  pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple || die "vLLM 安装失败"
  python -c "import vllm" || die "装完了但 import 失败，可能是 torch 版本冲突"
fi

# LoRA 支持检查：vLLM 从 0.2 起就支持，但定制镜像可能被裁掉
python - <<'PY' || die "当前 vLLM 不支持 LoRA（--enable-lora）"
import sys, vllm
try:
    from vllm.lora.request import LoRARequest  # noqa: F401
except Exception as e:
    print("LoRA 模块不可用:", e); sys.exit(1)
print("      LoRA 支持：OK")
PY

# ---- 3. 启动 ----
echo ""
echo "[3/4] 启动 vLLM（后台）"
echo "      日志：$LOG"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --enable-lora \
    --lora-modules "${SERVED_NAME}=${ADAPTER_DIR}" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "      pid $(cat "$PIDFILE")"

# ---- 4. 等就绪 ----
echo ""
echo "[4/4] 等待服务就绪（最多 5 分钟）……"
for i in $(seq 1 60); do
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo ""
    echo "      ❌ 进程已退出，最后 30 行日志："
    tail -30 "$LOG" | sed 's/^/      /'
    die "vLLM 启动失败"
  fi
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "      ✅ 服务就绪（用时约 $((i*5)) 秒）"
    break
  fi
  sleep 5
  [ $((i % 6)) -eq 0 ] && echo "      …已等 $((i*5)) 秒"
done

if ! curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
  echo ""
  echo "      超时。最后 30 行日志："
  tail -30 "$LOG" | sed 's/^/      /'
  die "vLLM 未在 5 分钟内就绪"
fi

echo ""
echo "=========================================================="
echo " 部署完成"
echo "=========================================================="
echo ""
echo "【1】看模型列表（应该有 $SERVED_NAME）"
echo "    curl -s http://127.0.0.1:$PORT/v1/models | python -m json.tool"
echo ""
echo "【2】跑一条真实问数（验证模型真的在答）"
echo "    curl -s http://127.0.0.1:$PORT/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"2月的销售额是多少？\"}],\"temperature\":0}' \\"
echo "      | python -m json.tool"
echo ""
echo "【3】从你自己的电脑访问 —— 把 AutoDL 的 6006 端口映射到本服务"
echo "    （AutoDL 控制台 → 自定义服务，或用 SSH 隧道："
echo "     ssh -p <端口> -L 8000:127.0.0.1:8000 root@<主机> ）"
echo "    然后本机 config 里 base_url 填 http://127.0.0.1:8000/v1"
echo ""
echo "【4】停服务"
echo "    kill \$(cat $PIDFILE)"
echo ""
echo "【磁盘】$(df -h "$BASE" | tail -1)"
