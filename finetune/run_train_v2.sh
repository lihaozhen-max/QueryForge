#!/bin/bash
# ============================================================================
# 掌柜问数 · 第二轮训练（v2 数据：修掉 JSON 列 bug + 补齐无实体超短问句）
#
# 用法（租卡机器，任意目录）：
#     bash /root/autodl-tmp/run_train_v2.sh
#
# 它会依次训练 ep1 / ep2 / ep3，每个约占 30 分钟（1591 条数据）。
# 训完再用 run_predict.sh 出预测：
#     bash /root/autodl-tmp/run_predict.sh          # 默认 LORA_TAG=v2
#
# 前提：/root/autodl-tmp/ 下已有
#     models/Qwen3-8B/                 基础模型
#     data/train_final.jsonl           1591 条（**必须上传 v2 版本**）
#     data/val_final.jsonl             138 条
#     data/dataset_info.json
#     train_ep1.yaml train_ep2.yaml train_ep3.yaml
# ============================================================================
set -u

BASE=/root/autodl-tmp
mkdir -p "$BASE/logs"
LOG="$BASE/logs/run_train_v2.log"
: > "$LOG"

say()  { echo "$*" | tee -a "$LOG"; }
step() { say ""; say "==================== $* ===================="; }
die()  { say ""; say "!!! 失败：$*"; say "日志：$LOG"; exit 1; }

say "开始：$(date '+%F %T')"
say "磁盘：$(df -h "$BASE" | tail -1)"

# ---- 前置检查 ----
step "0. 前置检查"
[ -f "$BASE/models/Qwen3-8B/config.json" ]  || die "缺少基础模型 $BASE/models/Qwen3-8B"
[ -f "$BASE/data/dataset_info.json" ]       || die "缺少 $BASE/data/dataset_info.json"
for f in train_final.jsonl val_final.jsonl; do
  [ -f "$BASE/data/$f" ] || die "缺少 $BASE/data/$f"
done
for ep in 1 2 3; do
  [ -f "$BASE/train_ep${ep}.yaml" ] || die "缺少 $BASE/train_ep${ep}.yaml"
done

N_TRAIN=$(wc -l < "$BASE/data/train_final.jsonl")
N_VAL=$(wc -l < "$BASE/data/val_final.jsonl")
say "  训练集 $N_TRAIN 条 / 验证集 $N_VAL 条"
if [ "$N_TRAIN" -lt 1400 ]; then
  say ""
  say "  ⚠️ 训练集只有 $N_TRAIN 条，预期是 1591 条 —— 你可能上传的还是**第一轮**的数据。"
  say "     第一轮是 656 条、澄清占比 10.7%；第二轮是 1591 条、澄清占比 15.2%。"
  say "     若确认要沿用旧数据，把下面这行的检查删掉再跑。"
  die "训练集条数不对，先确认上传的是 v2 数据"
fi

# ---- 训练 ----
for ep in 1 2 3; do
  OUT="$BASE/output/qwen3-8b-lora-v2-ep${ep}"
  if [ -f "$OUT/adapter_config.json" ]; then
    say ""
    say "ep${ep} 已有产出，跳过（如需重训，先删 $OUT）"
    continue
  fi
  step "${ep}. 训练 ${ep} epoch"
  llamafactory-cli train "$BASE/train_ep${ep}.yaml" > "$BASE/logs/train_v2_ep${ep}.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -30 "$BASE/logs/train_v2_ep${ep}.log" | tee -a "$LOG"
    die "ep${ep} 训练失败"
  fi
  grep -E "trainable params|Num examples|Total optimization steps" \
    "$BASE/logs/train_v2_ep${ep}.log" | sed 's/^/    /' | tee -a "$LOG"
  grep -E "'eval_loss'" "$BASE/logs/train_v2_ep${ep}.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
  say "    ep${ep} 完成：$(date '+%F %T')"
done

say ""
say "全部完成！$(date '+%F %T')"
say ""
say "下一步 —— 跑预测："
say "    bash $BASE/run_predict.sh"
say ""
say "【磁盘】$(df -h "$BASE" | tail -1)"
