#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本机 -> 租卡机器：把本地脚本内容通过 stdin 喂给远端 python 执行。

为什么不用 `ssh "python -c '...'"`：
    命令要穿过 PowerShell -> ssh -> 远端 bash 三层，
    里面的引号、中文、`{}`、`$` 会被反复解析，极易走形（已经踩过：
    一个 JSON 请求体被解析成 `{" model\\:\\qwen3-...}`，远端直接 500）。
    用 stdin 传文件内容就完全没有转义问题。

用法：
    python finetune/ssh_pipe.py 本地脚本.py [远端参数...]
    # 脚本内容照原样送给远端 `python - <参数>`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = os.getenv("DSH_SSH_HOST", "connect.westb.seetacloud.com")
PORT = int(os.getenv("DSH_SSH_PORT", "51756"))
USER = os.getenv("DSH_SSH_USER", "root")
PASSWORD = os.getenv("DSH_SSH_PASSWORD", "")


def run(local_script: Path, remote_args: list[str], timeout: int = 1800) -> int:
    if not PASSWORD:
        print("[错误] 未设置 DSH_SSH_PASSWORD", file=sys.stderr)
        return 2
    source = local_script.read_text(encoding="utf-8")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    try:
        # 远端非交互式 SSH **不会**加载 conda 的 init，所以 `python` 常常不在 PATH。
        # 用绝对路径最稳；可用 DSH_REMOTE_PYTHON 覆盖。
        py = os.getenv("DSH_REMOTE_PYTHON", "/root/miniconda3/bin/python")
        cmd = f"{py} - " + " ".join(f"'{a}'" for a in remote_args)
        print(f"\n$ {local_script.name} -> {cmd}", flush=True)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        stdin.write(source)
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            print(out.rstrip())
        if err.strip():
            print("[stderr]", err.rstrip())
        if rc != 0:
            print(f"[exit {rc}]")
        return rc
    finally:
        client.close()


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python ssh_pipe.py 本地脚本.py [远端参数...]", file=sys.stderr)
        return 2
    return run(Path(sys.argv[1]), sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
