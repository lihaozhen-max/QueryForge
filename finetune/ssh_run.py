#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过 SSH 连到租卡机器，执行一组命令并把输出带回来。

用法：
    python finetune/ssh_run.py "命令1" "命令2" ...

为什么单独写一个：本机没有 sshpass / plink，Windows OpenSSH 又需要一个真实 TTY
才能交互输密码，所以用 paramiko 在代码里把密码传进去，实现自动化。
密码通过环境变量 DSH_SSH_PASSWORD 传入，不写进代码。
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


def main() -> int:
    cmds = sys.argv[1:]
    if not cmds:
        print("用法: python ssh_run.py '命令' ['命令2' ...]", file=sys.stderr)
        return 2
    if not PASSWORD:
        print("[错误] 未设置 DSH_SSH_PASSWORD", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)

    rc_all = 0
    for cmd in cmds:
        print(f"\n$ {cmd}", flush=True)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=3600, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            print(out.rstrip())
        if err.strip():
            print("[stderr]", err.rstrip())
        if rc != 0:
            print(f"[exit {rc}]")
            rc_all = rc

    client.close()
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
