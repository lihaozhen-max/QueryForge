#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本机发一个问数请求给项目后端，并把 SSE 事件流打印出来。

为什么不用 curl：
    PowerShell 把中文参数按 **GBK** 传给 curl.exe，而服务端按 UTF-8 解析，
    于是报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd4`（500）。
    用 Python 发请求全程 UTF-8，不受 shell 编码影响。

用法：
    python finetune/query_once.py
    python finetune/query_once.py "广东省1月的销售额是多少？"
    python finetune/query_once.py --port 8001 "客单价怎么样"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="向项目后端发一个问数请求（SSE）")
    ap.add_argument("query", nargs="?", default="2月的销售额是多少？")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/api/query"
    payload = json.dumps({"query": args.query}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"})

    print(f"POST {url}")
    print(f"问题：{args.query}")
    print("-" * 64)
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            print(f"HTTP {resp.status}  Content-Type: {resp.headers.get('Content-Type')}")
            print("-" * 64)
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                # 尽量把 JSON 打印成可读形式；非 JSON 原样输出
                try:
                    obj = json.loads(line)
                    print(json.dumps(obj, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print(line)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}")
        print(e.read().decode("utf-8", "replace")[:2000])
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"请求失败：{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
