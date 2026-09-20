#!/bin/bash
# ============================================================================
# 掌柜问数 · 只跑预测（LoRA 版，不需要导出 16GB 合并模型）
#
# 用法（租卡机器，任意目录）：
#     LORA_TAG=v2-1ep bash /root/autodl-tmp/run_predict.sh
#     bash /root/autodl-tmp/run_predict.sh          # 默认 LORA_TAG=v2
#
# 环境变量（都可省略）：
#     LORA_TAG    LoRA 目录后缀 → output/qwen3-8b-lora-${LORA_TAG}-ep${N}
#                 只训了 1 epoch 的话是 v2-1ep；跑了三组消融则是 v2
#     EPOCHS      要预测哪几轮，默认 "1 2 3"（缺哪轮会自动跳过）
#     PRED_DIR    输出目录，默认 $BASE/preds_${LORA_TAG}
#
# 前提：已经有基础模型与 LoRA 权重。本脚本**不做训练、不做导出**。
#
# 为什么不用合并模型：
#   导出一次合并模型要 16GB，三个就 48GB，磁盘放不下（实测 No space left）。
#   直接在基础模型上挂 LoRA 即可，LoRA 只有几十 MB。
#
# 幂等：已存在的预测文件会跳过。**换了新数据集重训后，请换一个空目录
#       （或先删掉旧文件），否则会沿用上一轮的结果。**
#
# 输出文件命名固定为 base.jsonl / lora_ep{N}.jsonl（不带 tag），
# 这样本机的 run_eval_all.ps1 不用改就能直接评。
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
if [ ${#AVAIL[@]} -eq 0 ]; then
  say ""
  say "  在 output/ 下没找到 qwen3-8b-lora-${LORA_TAG}-ep*，实际存在的是："
  ls -d "$BASE"/output/qwen3-8b-lora-* 2>/dev/null | sed 's/^/    /' | tee -a "$LOG" || say "    （output 目录为空）"
  die "没有任何 LoRA 权重。检查 LORA_TAG 是否与训练时一致"
fi

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
