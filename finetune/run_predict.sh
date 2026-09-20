#!/bin/bash
# ============================================================================
# 掌柜问数 · 只跑预测（LoRA 版，不需要导出 16GB 合并模型）
#
# 用法（租卡机器，任意目录）：
#     bash /root/autodl-tmp/run_predict.sh
#
# 环境变量（都可省略，用默认值）：
#     LORA_TAG    LoRA 目录的后缀，默认 v2  → output/qwen3-8b-lora-${LORA_TAG}-ep1
#     PRED_DIR    预测输出目录，默认 $BASE/preds_${LORA_TAG}
#     例：LORA_TAG=v1 PRED_DIR=/root/autodl-tmp/preds bash run_predict.sh
#
# 前提：已经有基础模型与 LoRA 权重。本脚本**不做训练、不做导出**。
#
# 为什么不用合并模型：
#   导出一次合并模型要 16GB，三个就 48GB，磁盘放不下（实测 No space left）。
#   直接在基础模型上挂 LoRA 即可，LoRA 只有几十 MB。
#
# 幂等：已存在的预测文件会跳过。**换了新数据集重训后，请换一个空目录
#       （或先删掉旧文件），否则会沿用上一轮的结果。**
# ============================================================================
set -u

BASE=/root/autodl-tmp
EPOCHS="${EPOCHS:-1 2 3}"
LORA_TAG="${LORA_TAG:-v2}"
PRED_DIR="${PRED_DIR:-$BASE/preds_${LORA_TAG}}"

mkdir -p "$PRED_DIR" "$BASE/logs"
LOG="$BASE/logs/run_predict_${LORA_TAG}.log"
: > "$LOG"

say()  { echo "$*" | tee -a "$LOG"; }
step() { say ""; say "==================== $* ===================="; }
die()  { say ""; say "!!! 失败：$*"; say "日志：$LOG"; exit 1; }

say "开始：$(date '+%F %T')"
say "LoRA 组别：$LORA_TAG   预测输出：$PRED_DIR"
say "磁盘：$(df -h "$BASE" | tail -1)"

# ---- 前置检查 ----
step "0. 前置检查"
[ -f "$BASE/models/Qwen3-8B/config.json" ] || die "缺少基础模型 $BASE/models/Qwen3-8B"
[ -f "$BASE/predict_with_model.py" ]       || die "缺少 $BASE/predict_with_model.py"
[ -f "$BASE/data/eval_context.jsonl" ]     || die "缺少 $BASE/data/eval_context.jsonl"
say "  基础模型 / 预测脚本 / 评测上下文 就绪"

# 找出可用的 LoRA 权重
AVAIL=()
for ep in $EPOCHS; do
  if [ -f "$BASE/output/qwen3-8b-lora-${LORA_TAG}-ep${ep}/adapter_config.json" ]; then
    AVAIL+=("$ep")
    say "  [有] LoRA ep${ep} -> $BASE/output/qwen3-8b-lora-${LORA_TAG}-ep${ep}"
  else
    say "  [无] LoRA ep${ep}"
  fi
done
[ ${#AVAIL[@]} -gt 0 ] || die "没有任何 LoRA 权重，请先训练"

# ---- 原版模型 ----
if [ -s "$PRED_DIR/base.jsonl" ]; then
  say ""
  say "原版模型预测已存在，跳过（如需重跑，先删 $PRED_DIR/base.jsonl）"
else
  step "1. 预测：原版模型（未微调）"
  python "$BASE/predict_with_model.py" \
    --model "$BASE/models/Qwen3-8B" \
    --context "$BASE/data/eval_context.jsonl" \
    --out "$PRED_DIR/base.jsonl" > "$BASE/logs/pred_base.log" 2>&1
  [ $? -eq 0 ] || { tail -25 "$BASE/logs/pred_base.log" | tee -a "$LOG"; die "原版预测失败"; }
  grep -E "SQL .* 条" "$BASE/logs/pred_base.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
fi

# ---- 各轮 LoRA ----
for ep in "${AVAIL[@]}"; do
  OUT="$PRED_DIR/lora_ep${ep}.jsonl"
  if [ -s "$OUT" ]; then
    say ""
    say "微调 ${ep} 轮预测已存在，跳过（如需重跑，先删 $OUT）"
    continue
  fi
  step "2.${ep} 预测：微调模型（${ep} 轮，LoRA 挂载）"
  python "$BASE/predict_with_model.py" \
    --model "$BASE/models/Qwen3-8B" \
    --adapter "$BASE/output/qwen3-8b-lora-${LORA_TAG}-ep${ep}" \
    --context "$BASE/data/eval_context.jsonl" \
    --out "$OUT" > "$BASE/logs/pred_ep${ep}.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -25 "$BASE/logs/pred_ep${ep}.log" | tee -a "$LOG"
    die "微调 ${ep} 轮预测失败"
  fi
  grep -E "SQL .* 条" "$BASE/logs/pred_ep${ep}.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
done

# ---- 打包 ----
step "3. 打包结果"
cd "$PRED_DIR" || die "无法进入 $PRED_DIR"
FILES=$(ls base.jsonl lora_ep*.jsonl 2>/dev/null)
[ -n "$FILES" ] || die "没有预测文件"
tar czf "$BASE/result_preds_${LORA_TAG}.tar.gz" $FILES
say ""
say "完成！$(date '+%F %T')"
say ""
say "【下载清单】目录 $PRED_DIR/"
ls -l "$PRED_DIR"/*.jsonl | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【或直接下载压缩包】"
ls -lh "$BASE/result_preds_${LORA_TAG}.tar.gz" | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【磁盘】$(df -h "$BASE" | tail -1)"
