#!/bin/bash
# ============================================================================
# 掌柜问数 · 第二轮训练（v2 数据：修掉 JSON 列 bug + 补齐无实体超短问句）
#
# 用法（租卡机器，任意目录）：
#     bash /root/autodl-tmp/run_train_v2.sh
#
# 环境变量（都可省略）：
#     EPOCHS     要训练哪几组。**默认只训 1** —— 第一轮实测 1 epoch 是最优档，
#                所以默认走最短路径。想跑完整消融就：
#                    EPOCHS="1 2 3" bash run_train_v2.sh
#     RUN_TAG    产物标识，默认 v2
#
# 产物目录统一命名为：output/qwen3-8b-lora-<LORA_TAG>-ep<N>
#     只训 1 epoch  → LORA_TAG=v2-1ep → output/qwen3-8b-lora-v2-1ep-ep1
#     跑三组消融    → LORA_TAG=v2     → output/qwen3-8b-lora-v2-ep{1,2,3}
# 那个 -1ep 后缀是刻意的：避免"只跑 1 个 epoch 的产物"和
# "三组消融里的第 1 组"看起来一模一样。
#
# 训完出预测（LORA_TAG 用脚本最后打印的那个值）：
#     LORA_TAG=v2-1ep bash /root/autodl-tmp/run_predict.sh
#
# 时间预算（1591 条；第一轮 656 条时 1 epoch 约 10 分钟）
#     EPOCHS=1      约 30 分钟训练 + 10 分钟预测
#     EPOCHS=1 2 3  约 90–110 分钟训练
#
# 前提：/root/autodl-tmp/ 下已有
#     models/Qwen3-8B/                 基础模型
#     data/train_final.jsonl           1591 条（**必须上传 v2 版本**）
#     data/val_final.jsonl             138 条
#     data/dataset_info.json
#     train_ep1.yaml（只训 1 组时必须有；跑消融还要 train_ep2/3.yaml）
#     check_token_len.py               开跑前自检（可选，有就自动跑）
# ============================================================================
set -u

BASE=/root/autodl-tmp
EPOCHS="${EPOCHS:-1}"
RUN_TAG="${RUN_TAG:-v2}"

# 只训一组时给标识加 -1ep 后缀。
# 这一个变量同时决定：训练产物目录名 + 预测时 run_predict.sh 的 LORA_TAG。
if [ "$EPOCHS" = "1" ]; then
  LORA_TAG="${RUN_TAG}-1ep"
else
  LORA_TAG="${RUN_TAG}"
fi

mkdir -p "$BASE/logs"
LOG="$BASE/logs/run_train_v2.log"
: > "$LOG"

say()  { echo "$*" | tee -a "$LOG"; }
step() { say ""; say "==================== $* ===================="; }
die()  { say ""; say "!!! 失败：$*"; say "日志：$LOG"; exit 1; }

say "开始：$(date '+%F %T')"
say "训练组别：$EPOCHS   →   产物标识 LORA_TAG=$LORA_TAG"
say "磁盘：$(df -h "$BASE" | tail -1)"

# ---- 前置检查 ----
step "0. 前置检查"
[ -f "$BASE/models/Qwen3-8B/config.json" ]  || die "缺少基础模型 $BASE/models/Qwen3-8B"
[ -f "$BASE/data/dataset_info.json" ]       || die "缺少 $BASE/data/dataset_info.json"
for f in train_final.jsonl val_final.jsonl; do
  [ -f "$BASE/data/$f" ] || die "缺少 $BASE/data/$f"
done
for ep in $EPOCHS; do
  [ -f "$BASE/train_ep${ep}.yaml" ] || die "缺少 $BASE/train_ep${ep}.yaml"
done

N_TRAIN=$(wc -l < "$BASE/data/train_final.jsonl")
N_VAL=$(wc -l < "$BASE/data/val_final.jsonl")
say "  训练集 $N_TRAIN 条 / 验证集 $N_VAL 条"
if [ "$N_TRAIN" -lt 1400 ]; then
  say ""
  say "  ⚠️ 训练集只有 $N_TRAIN 条，预期是 1591 条 —— 你可能上传的还是**第一轮**的数据。"
  say "     第一轮 656 条 / 澄清占比 10.8%；第二轮 1591 条 / 澄清占比 15.0%。"
  say "     若确认要沿用旧数据，把这段检查删掉再跑。"
  die "训练集条数不对，先确认上传的是 v2 数据"
fi

# ---- 开跑前量 token 长度 ----
# 超长样本会被从右侧截断（正是【用户查询】与答案的位置），
# 训练照样跑完、loss 照样下降，但学的是残缺样本 —— 这类问题不报错，只能量。
if [ -f "$BASE/check_token_len.py" ]; then
  step "0.5 token 长度自检"
  CHK_OUT="$BASE/logs/token_len_check.log"
  python "$BASE/check_token_len.py" \
      --data "$BASE/data/train_final.jsonl" "$BASE/data/val_final.jsonl" > "$CHK_OUT" 2>&1
  CHK_RC=$?
  tail -4 "$CHK_OUT" | sed 's/^/    /' | tee -a "$LOG"
  if [ "$CHK_RC" -ne 0 ]; then
    say ""
    say "  ❌ token 自检未通过（退出码 $CHK_RC），明细见 $CHK_OUT"
    say "     请先处理超长样本（削 schema 或调大 cutoff_len），否则答案会被截掉。"
    die "token 长度自检未通过"
  fi
  say "  ✅ 无样本会被截断"
else
  say "  [提示] 没上传 check_token_len.py，跳过 token 自检（建议补上）"
fi

# ---- 训练 ----
for ep in $EPOCHS; do
  OUT="$BASE/output/qwen3-8b-lora-${LORA_TAG}-ep${ep}"
  if [ -f "$OUT/adapter_config.json" ]; then
    say ""
    say "ep${ep} 已有产出，跳过（如需重训，先删 $OUT）"
    continue
  fi
  step "${ep}. 训练 ${ep} epoch"
  say "  开始：$(date '+%F %T')"
  llamafactory-cli train "$BASE/train_ep${ep}.yaml" > "$BASE/logs/train_v2_ep${ep}.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -30 "$BASE/logs/train_v2_ep${ep}.log" | tee -a "$LOG"
    die "ep${ep} 训练失败"
  fi
  grep -E "trainable params|Num examples|Total optimization steps" \
    "$BASE/logs/train_v2_ep${ep}.log" | sed 's/^/    /' | tee -a "$LOG"
  grep -E "'eval_loss'" "$BASE/logs/train_v2_ep${ep}.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
  say "    产物：$OUT"
  say "    ep${ep} 完成：$(date '+%F %T')"
done

say ""
say "全部完成！$(date '+%F %T')"
say ""
say "▶ 下一步 —— 跑预测（LORA_TAG 必须用这个值）："
say ""
say "    LORA_TAG=${LORA_TAG} bash $BASE/run_predict.sh"
say ""
say "【磁盘】$(df -h "$BASE" | tail -1)"
