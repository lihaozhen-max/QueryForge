#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量训练样本的真实 token 长度，确认 cutoff_len 不会截断掉答案。

为什么需要它
------------
`train_ep*.yaml` 里 `cutoff_len: 4096`。训练样本的 human 段含完整 schema，
**如果整条样本超过 4096 token，LLaMA-Factory 会从右侧截断** ——
而 "右侧" 正是【用户查询】和标准答案所在的位置，样本会被悄悄毁掉，
训练照样跑完、loss 照样下降，但模型学的是残缺样本。

这类问题**不会报错**，只能靠量。所以每次改数据集/改 schema 长度后都要跑一遍。

两种加载 tokenizer 的方式（自动选）
----------------------------------
1. **`tokenizers` 库直接读 `tokenizer.json`**（首选，不需要 transformers）
   只要有一个 `tokenizer.json` 就能算，本地就能跑。
2. `transformers.AutoTokenizer`（从模型目录加载）

用法
----
    # 租卡机器：直接用基础模型目录里的 tokenizer
    python check_token_len.py

    # 只给 tokenizer.json 也行
    python check_token_len.py --tokenizer-json /path/to/tokenizer.json

    # 本机（没装 transformers、没下模型）也能跑：
    #   从 ModelScope 下载 tokenizer.json 后指过去
    python finetune/check_token_len.py --tokenizer-json finetune/data/_qwen_tok/tokenizer.json

输出
----
* 平均 / 中位 / p90 / p99 / 最大 token 数
* **超过 cutoff 的条数与占比**（>0 就说明有样本被截断）
* 按任务类型分组统计
* 最长几条的构成拆解（system / human / gpt 各占多少），便于定位该削哪里

结论怎么用
----------
* 超限条数为 0 或极少（<0.5%）-> 可以开训
* 超限较多 -> 三选一：① 调大 cutoff_len（显存够的话）；
  ② 减少 schema 里的 examples 数量；③ 砍掉冗余的列描述
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        print(f"[跳过] 找不到 {p}")
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class _Tok:
    """统一封装，只暴露"把字符串变成 token 数"这一件事。"""

    def __init__(self, fn, label: str):
        self._fn = fn
        self.label = label

    def n(self, text: str) -> int:
        return self._fn(text)


def build_tokenizer(args) -> _Tok:
    # ---- 方式 1：tokenizers 直接读 tokenizer.json ----
    json_path = args.tokenizer_json
    if json_path is None:
        cand = Path(args.tokenizer) / "tokenizer.json"
        if cand.exists():
            json_path = cand

    if json_path is not None and Path(json_path).exists():
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(str(json_path))

            def fn(text: str) -> int:
                return len(tk.encode(text, add_special_tokens=False).ids)

            return _Tok(fn, f"tokenizers（{json_path}，词表 {tk.get_vocab_size()}）")
        except ImportError:
            print("[提示] 没装 tokenizers，改用 transformers")

    # ---- 方式 2：transformers ----
    from transformers import AutoTokenizer
    tk2 = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def fn2(text: str) -> int:
        return len(tk2.encode(text, add_special_tokens=False))

    return _Tok(fn2, f"transformers（{args.tokenizer}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="量训练样本的 token 长度，检查是否会被截断")
    ap.add_argument("--data", type=Path, nargs="+",
                    default=[Path("data/train_final.jsonl"), Path("data/val_final.jsonl")])
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3-8B",
                    help="模型目录（里面有 tokenizer.json 时优先用它）")
    ap.add_argument("--tokenizer-json", type=Path, default=None,
                    help="直接指定 tokenizer.json（本机没下模型时用）")
    ap.add_argument("--cutoff", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=5, help="打印几条最长的做拆解")
    args = ap.parse_args()

    tok = build_tokenizer(args)
    print(f"tokenizer : {tok.label}")
    print(f"cutoff_len: {args.cutoff}")
    print()

    grand_over = 0
    grand_total = 0
    for path in args.data:
        recs = load_jsonl(path)
        if not recs:
            continue
        print("=" * 72)
        print(f"{path}   共 {len(recs)} 条")
        print("=" * 72)

        lens: list[int] = []
        parts: list[tuple[int, int, int]] = []
        over: list[tuple[int, str, str, int]] = []
        by_task: dict[str, list[int]] = {}

        for r in recs:
            c = r.get("conversations") or []
            if len(c) < 3:
                continue
            ns = tok.n(c[0]["value"])
            nh = tok.n(c[1]["value"])
            ng = tok.n(c[2]["value"])
            n = ns + nh + ng
            lens.append(n)
            parts.append((ns, nh, ng))
            task = r.get("task", "?")
            by_task.setdefault(task, []).append(n)
            if n > args.cutoff:
                over.append((n, task, r.get("question", "")[:40], len(lens) - 1))

        if not lens:
            continue

        s = sorted(lens)
        print(f"  token 长度：平均 {statistics.mean(lens):.0f} | 中位 {s[len(s)//2]} | "
              f"p90 {s[int(len(s)*0.90)]} | p99 {s[int(len(s)*0.99)]} | 最大 {max(s)}")
        print(f"  构成占比：system {statistics.mean(p[0] for p in parts):.0f} | "
              f"human(schema+问题) {statistics.mean(p[1] for p in parts):.0f} | "
              f"gpt(答案) {statistics.mean(p[2] for p in parts):.0f}")
        print()
        print(f"  >>> 超过 cutoff({args.cutoff}) 的样本：{len(over)} 条 "
              f"({len(over)/len(lens)*100:.2f}%)")
        if over:
            print("      ⚠️ 这些样本会被右侧截断，【用户查询】/答案可能整段丢失！")
            cnt: dict[str, int] = {}
            for _, t, _q, _i in over:
                cnt[t] = cnt.get(t, 0) + 1
            for t, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
                tot = len(by_task.get(t, []))
                print(f"        {t:<14} {v:>4} / {tot:<4} 条  ({v/max(1,tot)*100:.0f}%)")
        print()

        print("  按任务类型：")
        for t, arr in sorted(by_task.items()):
            print(f"    {t:<14} n={len(arr):<5} 平均 {statistics.mean(arr):>5.0f}  最大 {max(arr):>5}")
        print()

        if args.sample and over:
            print(f"  最长的 {args.sample} 条拆解：")
            for n, t, q, idx in sorted(over, reverse=True)[: args.sample]:
                ns, nh, ng = parts[idx]
                print(f"    {n:>5} tok  [{t}] {q}")
                print(f"           system={ns}  human={nh}  gpt={ng}")
                if ng > 300:
                    print("           → gpt（答案）本身太长")
                elif nh > args.cutoff * 0.8:
                    print("           → human（schema）占绝大部分，该削 schema")
            print()

        grand_over += len(over)
        grand_total += len(lens)

    print("=" * 72)
    if grand_total:
        print(f"总计：{grand_total} 条样本，超限 {grand_over} 条 "
              f"({grand_over/grand_total*100:.2f}%)")
        if grand_over == 0:
            print("✅ 无样本会被截断，可以开训。")
            return 0
        print("❌ 存在被截断的样本，先处理再训（见文件头「结论怎么用」）。")
        return 1
    print("没有读到任何样本。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
