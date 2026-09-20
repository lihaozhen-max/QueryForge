#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在租卡机器上用指定模型对评测集做预测（生成 SQL 或澄清/拒答）。

用途
----
对「原版 Qwen3-8B」与「微调后的 LoRA 适配层」各跑一次，产出多份预测文件，
下载回本机后用 eval_ex.py 算 EX，填充对比矩阵。

**不需要合并模型**：用 `--adapter` 指定 LoRA 目录，脚本会自己
`PeftModel.from_pretrained(...) + merge_and_unload()`。
早期版本要求先 export 出 16GB 合并模型，三组实验就是 48GB，
本机（50GB 磁盘）会在导出时 `No space left on device`，所以已改成适配层直载。

输入
----
eval_context.jsonl  由本机 dump_context.py 导出，含每题的 schema 上下文。
                    （租卡机器上没有 MySQL/Qdrant/ES，无法现场检索，故预先生成）

输出
----
预测文件，每行 {"id", "type", "content"}，可直接喂给 eval_ex.py。

要点
----
* **不依赖 LLaMA-Factory**：只用 transformers，避免版本/CLI 差异。
* **bf16 加载**（你的"16 精度"），单卡即可，8B 模型约 16G 显存。
* **system 提示词与训练时完全一致**：训练数据里已内嵌该 system，
  推理时复用同一份，保证同分布 —— 这样模型才会在信息不足时输出澄清。
* 支持 `--limit` 冒烟与 `--resume` 断点续跑。

用法
----
    # 冒烟 5 条
    python predict_with_model.py \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --context /root/autodl-tmp/data/eval_context.jsonl \
        --out /root/autodl-tmp/preds/base.jsonl --limit 5

    # 原版模型全量
    python predict_with_model.py \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --context /root/autodl-tmp/data/eval_context.jsonl \
        --out /root/autodl-tmp/preds/base.jsonl

    # 微调后模型全量（--adapter 指向 LoRA 目录，无需先合并出 16GB 完整模型）
    python predict_with_model.py \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --adapter /root/autodl-tmp/output/qwen3-8b-lora-ep1 \
        --context /root/autodl-tmp/data/eval_context.jsonl \
        --out /root/autodl-tmp/preds/lora_ep1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# --------------------------------------------------------------------------
# system 提示词：与 finetune/build_final_train_set.py 的 UNIFIED_SYSTEM 完全一致
# 训练数据里已内嵌这段，推理必须复用，否则分布不一致（模型会不认识这个任务）。
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """【角色】
你是一个资深的数据库专家和数据分析师。你的任务是根据提供的【上下文信息】，将用户的自然语言查询转换为语法正确、性能优化的 SQL 语句。
【上下文信息】
可用数据表信息如下：
可参考的指标信息如下：
当前的时间信息如下：
数据库环境如下：
【任务要求】
1. 仅允许使用数据表信息中真实存在的表与字段名称，禁止编造、猜测或引入未提供的表和字段。
2. 若指标信息中存在相关指标定义，必须严格遵循其业务口径、计算逻辑、过滤规则与时间口径，不得偏离或重解释，若指标信息未覆盖用户问题中的指标或口径，则基于用户问题语义与通用业务常识进行。
3. 生成的SQL只能用于查询，不能涉及数据写入、更新、删除等操作。
4. 生成的SQL的语法必须严格符合数据库环境中指定的数据库类型与版本。
5. 默认只生成一条SQL，不可生成多条SQL。
6. 输出必须仅包含一条完整 SQL 语句的纯文本，严禁使用```、```sql 等 Markdown代码块或任何格式化符号。

【输出类型】
你必须二选一输出：
A. 若问题已具备「时间范围」「统计粒度」「指标口径」等必要信息，输出一条 MySQL 只读查询语句；
B. 若信息不足，不要臆测，改为提出澄清问题：先用一句话指出缺少哪些信息，再列出 2-3 个具体可选的确认项，每个确认项给出可选值，便于用户直接回答。
无论哪种情况，都只输出纯文本，不要使用 Markdown 代码块。"""

_SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def build_human(rec: dict) -> str:
    import yaml

    def dump(o) -> str:
        return yaml.dump(o, allow_unicode=True, sort_keys=False).strip() if o else "（无）"

    return (
        "【可用数据表信息】\n" + dump(rec.get("table_infos")) + "\n\n"
        "【可参考的指标信息】\n" + dump(rec.get("metric_infos")) + "\n\n"
        "【当前时间信息】\n" + dump(rec.get("date_info")) + "\n\n"
        "【数据库环境】\n" + dump(rec.get("db_info")) + "\n\n"
        "【用户查询】\n" + rec["question"]
    )


def clean_output(text: str, thinking: bool = True) -> str:
    """
    去掉思维链与 markdown 围栏，只留最终答案。

    思维链的分隔符有**两种**写法，必须都处理：

    * 普通 ASCII：`<think> ... </think>`
      —— chat_template 在 `enable_thinking=False` 时**预先塞进提示词**的
         `<think>\\n\\n</think>\\n\\n` 用的就是这种。
    * Qwen 专用 Unicode：`<think> ... <｜end▁of▁thinking｜>`
      —— 模型自己**生成**思维链时用的（`｜` 是全角竖线、`▁` 是 U+2581）。

    早期版本只匹配第二种，于是第一种一个字符都剥不掉；而那时恰好关着思维链，
    模型就是"从空思维链块接着写"，剥不掉也没暴露问题。
    改成两种都剥，并在剥完后兜底再查一次残留的结束标记。

    `thinking=False` 时不做任何思维链剥离 —— 那种条件下提示词里已经带了空思维链块，
    模型直接续写答案，文本里不会有思维链标记；此时任何剥离都是多余且可能误伤的。
    """
    t = (text or "").strip()
    if thinking:
        # 显式思维链（两种结束标记都覆盖）
        t = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<think\b[^>]*>.*?<｜end▁of▁thinking｜>", "", t, flags=re.DOTALL | re.IGNORECASE)
        # 兜底：不成对时，从开头到结束标记整段丢掉
        if "<think" in t.lower():
            t = re.sub(r"^.*?(?:</think\s*>|<｜end▁of▁thinking｜>)", "", t, flags=re.DOTALL | re.IGNORECASE)
        # 残留的孤立标记
        t = re.sub(r"</?think\b[^>]*>", "", t, flags=re.IGNORECASE)
        t = t.replace("<｜end▁of▁thinking｜>", "")
    # markdown 围栏
    t = re.sub(r"^```(?:sql)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    return t.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="用指定模型对评测集做预测")
    ap.add_argument("--model", required=True, help="基础模型目录")
    ap.add_argument("--adapter", default="",
                    help="LoRA 适配器目录（可选）。给了就在基础模型上挂 LoRA，"
                         "**不需要先导出 16GB 的合并模型**，省磁盘也省时间。")
    ap.add_argument("--context", required=True, help="eval_context.jsonl")
    ap.add_argument("--out", required=True, help="预测输出 jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    ap.add_argument("--resume", action="store_true", help="跳过已预测的")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    # ---- 思维链开关 ----
    # 这里原先写的是 `--no-think` + `action="store_true", default=True`，
    # 那是个 **bug**：store_true 配 default=True 意味着 args.no_think 恒为 True，
    # 于是 enable_thinking 恒为 False —— 这个开关**永远打不开思维链**。
    #
    # 为什么必须能打开：Qwen3 的 chat_template 里
    #   enable_thinking=False → 渲染成 '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    #                            （预先塞一个**空**思维链块，等于命令模型"别想，直接答"）
    #   enable_thinking=True  → 渲染成 '<|im_start|>assistant\n'（让模型自己决定要不要想）
    # 二者是**不同的输入**，不是同一件事的开关。
    # 实测：关掉思维链时模型会退化成硬编码日期（BETWEEN 20250201 AND 20250228），
    #       开着时会走维表（IN (SELECT date_id FROM dim_date WHERE month = 2)）。
    # 训练用的是 LLaMA-Factory 的 `qwen3` 模板，是否等价于 False 需以实际渲染为准，
    # 因此默认值设为 True（让模型自己决定），并用 --no-think 才能关掉。
    ap.add_argument("--think", dest="enable_thinking", action="store_true", default=True,
                    help="开启思维链（默认）。与训练条件保持一致")
    ap.add_argument("--no-think", dest="enable_thinking", action="store_false",
                    help="关闭思维链。注意：这会让输入多一个空  thinking 块，"
                         "实测 SQL 质量会下降，除非确认训练时也是这么喂的")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ctx_path = Path(args.context)
    if not ctx_path.exists():
        print(f"[错误] 找不到上下文文件 {ctx_path}", file=sys.stderr)
        return 2

    records = [json.loads(l) for l in ctx_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    done: set[str] = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        print("[完成] 无待预测样本")
        return 0

    print(f"[1/3] 待预测 {len(todo)} 条（共 {len(records)}）")
    print(f"[2/3] 加载模型：{args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if args.adapter:
        # 直接在基础模型上挂 LoRA，避免导出 16GB 合并模型（磁盘吃不消）
        from peft import PeftModel
        print(f"      挂载 LoRA 适配器：{args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # 合并进权重，推理更快
    model.eval()
    print(f"      加载完成（{time.time()-t0:.0f}s），dtype={next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        print(f"      显存占用 {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sql = n_clar = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_human(rec)},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,          # 预测要确定性，不用采样
                    repetition_penalty=1.05,
                )
            out_ids = gen[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(out_ids, skip_special_tokens=True)
            content = clean_output(raw, thinking=args.enable_thinking)
            ptype = "SQL" if _SELECT_RE.match(content) else "CLARIFY"
            if ptype == "SQL":
                n_sql += 1
            else:
                n_clar += 1

            fh.write(json.dumps({
                "id": rec["id"], "question": rec["question"],
                "type": ptype, "content": content,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i <= 3:
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] SQL={n_sql} 澄清/拒答={n_clar} "
                      f"({el:.0f}s, {el/i:.1f}s/条)")

    print()
    print(f"[3/3] 完成：{len(todo)} 条，用时 {time.time()-t0:.0f}s")
    print(f"      SQL {n_sql} 条 / 澄清或拒答 {n_clar} 条")
    print(f"输出 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
