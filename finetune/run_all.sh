#!/bin/bash
# ============================================================================
# 掌柜问数 · 微调一键脚本（全自动，路径全部内置，不需要你输任何路径）
#
# 用法（在租卡机器上，任意目录都行）：
#     bash run_all.sh
#     bash run_all.sh 2>&1 | tee /root/autodl-tmp/run.log   # 同时存日志
#
# 依次完成：
#   0. 自检（数据能否被 pyarrow 读、模型/配置/脚本是否齐全）
#   1. 训练 1 / 2 / 3 轮（轮数消融）
#   2. 导出三个合并模型
#   3. 预测：原版模型 + 三个微调模型
#   4. 打包结果 + 打印下载清单
#
# 出错会立刻停止并打印日志尾部，不会闷头跑。
# ============================================================================
set -u

BASE=/root/autodl-tmp
mkdir -p "$BASE/logs" "$BASE/preds"
LOG="$BASE/logs/run_all.log"
: > "$LOG"

say()  { echo "$*" | tee -a "$LOG"; }
step() { say ""; say "==================== $* ===================="; }
die()  { say ""; say "!!! 失败：$*"; say "详细日志：$LOG"; exit 1; }

say "开始时间：$(date '+%F %T')"
say "工作目录：$BASE"

# ---------------------------------------------------------------- 0. 自检
step "0. 自检"

say "[数据] 校验 jsonl 能否被 pyarrow 读取（LLaMA-Factory 用的解析器）"
python - <<'PY' 2>&1 | tee -a "$LOG"
import sys
import pyarrow.json as pj
ok = True
for n in ("train_final", "val_final"):
    p = f"/root/autodl-tmp/data/{n}.jsonl"
    try:
        t = pj.read_json(p)
        print(f"    [OK] {n}.jsonl -> {t.num_rows} 行")
    except Exception as e:
        ok = False
        print(f"    [FAIL] {n}.jsonl -> {str(e)[:200]}")
sys.exit(0 if ok else 1)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || die "训练数据无法被 pyarrow 读取，请重新上传 data/train_final.jsonl 与 data/val_final.jsonl"

say "[文件] 检查必需文件"
need_files=(
  "models/Qwen3-8B/config.json"
  "data/train_final.jsonl"
  "data/val_final.jsonl"
  "data/dataset_info.json"
  "data/eval_context.jsonl"
  "predict_with_model.py"
  "train_ep1.yaml"
  "train_ep2.yaml"
  "train_ep3.yaml"
  "export_qwen3_8b.yaml"
)
miss=0
for f in "${need_files[@]}"; do
  if [ -e "$BASE/$f" ]; then
    echo "    [OK]   $f" | tee -a "$LOG"
  else
    echo "    [缺]   $f" | tee -a "$LOG"
    miss=1
  fi
done
[ $miss -eq 0 ] || die "缺少上面的文件，请补齐后重跑"

grep -q "zhangguiwenshu_sft" "$BASE/data/dataset_info.json" \
  || die "data/dataset_info.json 里没有 zhangguiwenshu_sft 注册项"
say "[OK] 自检通过，开始训练"

# ---------------------------------------------------------------- 1. 训练
for ep in 1 2 3; do
  step "1.$ep 训练 ${ep} 轮"
  t0=$(date +%s)
  llamafactory-cli train "$BASE/train_ep${ep}.yaml" > "$BASE/logs/train_ep${ep}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "  训练失败，日志尾部："
    tail -25 "$BASE/logs/train_ep${ep}.log" | tee -a "$LOG"
    die "训练 ${ep} 轮失败"
  fi
  grep -E "trainable params|Num examples|Total optimization steps" "$BASE/logs/train_ep${ep}.log" \
    | tail -3 | sed 's/^/    /' | tee -a "$LOG"
  grep -E "eval_loss" "$BASE/logs/train_ep${ep}.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
  say "  完成，用时 $(( $(date +%s) - t0 )) 秒"
done

# ---------------------------------------------------------------- 2. 导出
for ep in 1 2 3; do
  step "2.$ep 导出合并模型（${ep} 轮）"
  cat > "$BASE/export_ep${ep}.yaml" <<EOF
model_name_or_path: $BASE/models/Qwen3-8B
adapter_name_or_path: $BASE/output/qwen3-8b-lora-ep${ep}
template: qwen3
finetuning_type: lora
export_dir: $BASE/output/qwen3-8b-merged-ep${ep}
export_size: 4
export_device: cpu
export_legacy_format: false
EOF
  llamafactory-cli export "$BASE/export_ep${ep}.yaml" > "$BASE/logs/export_ep${ep}.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -25 "$BASE/logs/export_ep${ep}.log" | tee -a "$LOG"
    die "导出 ${ep} 轮模型失败"
  fi
  say "  已导出 -> $BASE/output/qwen3-8b-merged-ep${ep}"
done

# ---------------------------------------------------------------- 3. 预测
step "3.0 预测：原版模型（未微调）"
python "$BASE/predict_with_model.py" \
  --model "$BASE/models/Qwen3-8B" \
  --context "$BASE/data/eval_context.jsonl" \
  --out "$BASE/preds/base.jsonl" > "$BASE/logs/pred_base.log" 2>&1
if [ $? -ne 0 ]; then
  tail -25 "$BASE/logs/pred_base.log" | tee -a "$LOG"
  die "原版模型预测失败"
fi
grep -E "SQL .* 条" "$BASE/logs/pred_base.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"

for ep in 1 2 3; do
  step "3.$ep 预测：微调模型（${ep} 轮）"
  python "$BASE/predict_with_model.py" \
    --model "$BASE/output/qwen3-8b-merged-ep${ep}" \
    --context "$BASE/data/eval_context.jsonl" \
    --out "$BASE/preds/lora_ep${ep}.jsonl" > "$BASE/logs/pred_ep${ep}.log" 2>&1
  if [ $? -ne 0 ]; then
    tail -25 "$BASE/logs/pred_ep${ep}.log" | tee -a "$LOG"
    die "微调模型（${ep} 轮）预测失败"
  fi
  grep -E "SQL .* 条" "$BASE/logs/pred_ep${ep}.log" | tail -1 | sed 's/^/    /' | tee -a "$LOG"
done

# ---------------------------------------------------------------- 4. 打包
step "4. 打包与下载清单"
cd "$BASE/preds" || die "无法进入 $BASE/preds"
tar czf "$BASE/result_preds.tar.gz" base.jsonl lora_ep1.jsonl lora_ep2.jsonl lora_ep3.jsonl
say ""
say "全部完成！结束时间：$(date '+%F %T')"
say ""
say "【要下载的文件】目录：$BASE/preds/"
ls -l "$BASE/preds/"*.jsonl | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【更省事】直接下载这个压缩包（含上面 4 个文件）："
ls -lh "$BASE/result_preds.tar.gz" | sed 's/^/    /' | tee -a "$LOG"
say ""
say "【备份 LoRA】如需换机器，可打包 LoRA 权重（很小）："
say "    cd $BASE && tar czf loara_backup.tar.gz output/qwen3-8b-lora-ep1 output/qwen3-8b-lora-ep2 output/qwen3-8b-lora-ep3"
