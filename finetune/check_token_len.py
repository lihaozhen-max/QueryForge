#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在租卡机器上量训练样本的真实 token 长度，确认 cutoff_len 不会截断掉答案。

为什么需要它
------------
`train_ep*.yaml` 里 `cutoff_len: 4096`。训练样本的 human 段含完整 schema，
**如果整条样本超过 4096 token，LLaMA-Factory 会从右侧截断** ——
而 "右侧" 正是【用户查询】和标准答案所在的位置，样本会被悄悄毁掉，
训练照样跑完、loss 照样下降，但模型学的是残缺样本。

这类问题**不会报错**，只能靠量。所以每次改数据集/改 schema 长度后都要跑一遍。

用法（租卡机器，在 /root/autodl-tmp 下）
---------------------------------------
    python check_token_len.py
    python check_token_len.py --cutoff 4096 --data data/train_final.jsonl data/val_final.jsonl
    python check_token_len.py --sample 8      # 多打印几条最长的看长什么样

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


def main() -> int:
    ap = argparse.ArgumentParser(description="量训练样本的 token 长度，检查是否会被截断")
    ap.add_argument("--data", type=Path, nargs="+",
                    default=[Path("data/train_final.jsonl"), Path("data/val_final.jsonl")])
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen3-8B",
                    help="tokenizer 路径（默认直接用基础模型目录）")
    ap.add_argument("--cutoff", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=5, help="打印几条最长的做拆解")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"tokenizer: {args.tokenizer}")
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
        over: list[tuple[int, str, str]] = []
        by_task: dict[str, list[int]] = {}

        for r in recs:
            c = r.get("conversations") or []
            if len(c) < 3:
                continue
            ns = len(tk.encode(c[0]["value"], add_special_tokens=False))
            nh = len(tk.encode(c[1]["value"], add_special_tokens=False))
            ng = len(tk.encode(c[2]["value"], add_special_tokens=False))
            n = ns + nh + ng
            lens.append(n)
            parts.append((ns, nh, ng))
            task = r.get("task", "?")
            by_task.setdefault(task, []).append(n)
            if n > args.cutoff:
                over.append((n, task, r.get("question", "")[:40]))

        if not lens:
            continue

        s = sorted(lens)
        print(f"  token 长度：平均 {statistics.mean(lens):.0f} | 中位 {s[len(s)//2]} | "
              f"p90 {s[int(len(s)*0.90)]} | p99 {s[int(len(s)*0.99)]} | 最大 {max(s)}")
        ns_avg = statistics.mean(p[0] for p in parts)
        nh_avg = statistics.mean(p[1] for p in parts)
        ng_avg = statistics.mean(p[2] for p in parts)
        print(f"  构成占比：system {ns_avg:.0f} | human(schema+问题) {nh_avg:.0f} | gpt(答案) {ng_avg:.0f}")
        print()
        print(f"  >>> 超过 cutoff({args.cutoff}) 的样本：{len(over)} 条 "
              f"({len(over)/len(lens)*100:.2f}%)")
        if over:
            print("      ⚠️ 这些样本会被右侧截断，【用户查询】/答案可能整段丢失！")
            print("      按任务类型：")
            cnt: dict[str, int] = {}
            for _, t, _q in over:
                cnt[t] = cnt.get(t, 0) + 1
            for t, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
                tot = len(by_task.get(t, []))
                print(f"        {t:<14} {v:>4} / {tot:<4} 条  ({v/max(1,tot)*100:.0f}%)")
        print()

        print("  按任务类型：")
        for t, arr in sorted(by_task.items()):
            a = sorted(arr)
            print(f"    {t:<14} n={len(a):<5} 平均 {statistics.mean(a):>5.0f}  最大 {max(a):>5}")
        print()

        if args.sample and over:
            print(f"  最长的 {args.sample} 条拆解：")
            for n, t, q in sorted(over, reverse=True)[: args.sample]:
                print(f"    {n:>5} tok  [{t}] {q}")
                idx = lens.index(n)
                ns, nh, ng = parts[idx]
                print(f"           system={ns}  human={nh}  gpt={ng}")
                if ng > 200:
                    print("           → gpt 段偏长，答案本身太长")
                elif nh > args.cutoff * 0.8:
                    print("           → human 段（schema）占绝大部分，该削 schema")
            print()

        grand_over += len(over)
        grand_total += len(lens)

    print("=" * 72)
    if grand_total:
        print(f"总计：{grand_total} 条样本，超限 {grand_over} 条 "
              f"({grand_over/grand_total*100:.2f}%)")
        if grand_over == 0:
            print("✅ 无样本会被截断，可以开训。")
        else:
            print("❌ 存在被截断的样本，先处理再训（见文件头「结论怎么用」）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
