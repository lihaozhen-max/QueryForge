#!/bin/bash
# ============================================================================
# 掌柜问数 · 只跑预测（LoRA 版，不需要导出 16GB 合并模型）
#
# 用法（租卡机器，任意目录）：
#     bash /root/autodl-tmp/run_predict.sh
#
# 前提：已经有基础模型与 LoRA 权重。本脚本**不做训练、不做导出**。
#
# 为什么不用合并模型：
#   导出一次合并模型要 16GB，三个就 48GB，磁盘放不下（实测 No space left）。
#   直接在基础模型上挂 LoRA 即可，LoRA 只有几十 MB。
# ============================================================================
set -u

BASE=/root/autodl-tmp
mkdir -p "$BASE/preds" "$BASE/logs"
LOG="$BASE/logs/run_predict.log"
: > "$LOG"

say()  { echo "$*" | tee -a "$LOG"; }
step() { say ""; say "==================== $* ===================="; }
die()  { say ""; say "!!! 失败：$*"; say "日志：$LOG"; exit 1; }

say "开始：$(date '+%F %T')"
say "磁盘：$(df -h "$BASE" | tail -1)"

# ---- 前置检查 ----
step "0. 前置检查"
[ -f "$BASE/models/Qwen3-8B/config.json" ] || die "缺少基础模型 $BASE/models/Qwen3-8B"
[ -f "$BASE/predict_with_model.py" ]       || die "缺少 $BASE/predict_with_model.py"
[ -f "$BASE/data/eval_context.jsonl" ]     || die "缺少 $BASE/data/eval_context.jsonl"
say "  基础模型 / 预测脚本 / 评测上下文 就绪"

# 找出可用的 LoRA 权重
AVAIL=()
for ep in 1 2 3; do
  if [ -f "$BASE/output/qwen3-8b-lora-ep${ep}/adapter_config.json" ]; then
    AVAIL+=("$ep")
    say "  [有] LoRA ep${ep} -> $BASE/output/qwen3-8b-lora-ep${ep}"
  else
    say "  [无] LoRA ep${ep}"
  fi
done
[ ${#AVAIL[@]} -gt 0 ] || die "没有任何 LoRA 权重，请先训练"

# ---- 原版模型 ----
if [ -s "$BASE/preds/base.jsonl" ]; then
  say ""
  say "原版模型预测已存在，跳过（如需重跑，先删 $BASE/preds/base.jsonl）"
else
  step "1. 预测：原版模型（未微调）"
  python "$BASE/predict_with_model.py" \
    --model "$BASE/models/Qwen3-8B" \
    --context "$BASE/data/eval_context.jsonl" \
    --out "$BASE/preds/base.jsonl" > "$BASE/logs/pred_base.log" 2>&1
  [ $? -eq 0 ] || { tail -25 "$BASE/logs/pred_base.log" | tee -a "$LOG"; die "原版预测失败"; }
  grep -E "SQL .* 条" "$BASE/logs/pred_base.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
fi

# ---- 各轮 LoRA ----
for ep in "${AVAIL[@]}"; do
  OUT="$BASE/preds/lora_ep${ep}.jsonl"
  if [ -s "$OUT" ]; then
    say ""
    say "微调 ${ep} 轮预测已存在，跳过（如需重跑，先删 $OUT）"
    continue
  fi
  step "2.${ep} 预测：微调模型（${ep} 轮，LoRA 挂载）"
  python "$BASE/predict_with_model.py" \
    --model "$BASE/models/Qwen3-8B" \
    --adapter "$BASE/output/qwen3-8b-lora-ep${ep}" \
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
cd "$BASE/preds" || die "无法进入 preds"
FILES=$(ls base.jsonl lora_ep*.jsonl 2>/dev/null)
[ -n "$FILES" ] || die "没有预测文件"
tar czf "$BASE/result_preds.tar.gz" $FILES
say ""
say "完成！$(date '+%F %T')"
say ""
say "【下载清单】目录 $BASE/preds/"
ls -l "$BASE/preds/"*.jsonl | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【或直接下载压缩包】"
ls -lh "$BASE/result_preds.tar.gz" | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【磁盘】$(df -h "$BASE" | tail -1)"
