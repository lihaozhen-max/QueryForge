#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本机 <-> 租卡机器的 SSH 端口转发（相当于 `ssh -L`）。

为什么需要：模型服务跑在租卡机器的 127.0.0.1:8000，外网访问不到。
把远端 8000 映射到本机 8000，本机项目就能用 `http://127.0.0.1:8000/v1` 访问微调模型，
业务代码里不需要填任何隧道/内网地址。

用法（前台运行，Ctrl+C 结束）：
    python finetune/ssh_tunnel.py
    python finetune/ssh_tunnel.py --local-port 8000 --remote-port 8000
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import socketserver
import sys
import threading
import time

import paramiko

HOST = os.getenv("DSH_SSH_HOST", "connect.westb.seetacloud.com")
PORT = int(os.getenv("DSH_SSH_PORT", "51756"))
USER = os.getenv("DSH_SSH_USER", "root")
PASSWORD = os.getenv("DSH_SSH_PASSWORD", "")


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            chan = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.chain_host, self.chain_port),
                self.request.getpeername(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[tunnel] 打开通道失败: {exc}", file=sys.stderr, flush=True)
            return
        if chan is None:
            print("[tunnel] 通道为空", file=sys.stderr, flush=True)
            return
        try:
            while True:
                r, _, _ = select.select([self.request, chan], [], [])
                if self.request in r:
                    data = self.request.recv(4096)
                    if not data:
                        break
                    chan.send(data)
                if chan in r:
                    data = chan.recv(4096)
                    if not data:
                        break
                    self.request.send(data)
        except Exception:  # noqa: BLE001
            pass
        finally:
            chan.close()
            self.request.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="SSH 端口转发到租卡机器")
    ap.add_argument("--local-port", type=int, default=8000)
    ap.add_argument("--remote-port", type=int, default=8000)
    ap.add_argument("--remote-host", default="127.0.0.1")
    args = ap.parse_args()

    if not PASSWORD:
        print("[错误] 未设置 DSH_SSH_PASSWORD", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)

    transport = client.get_transport()
    transport.set_keepalive(30)

    Handler.ssh_transport = transport
    Handler.chain_host = args.remote_host
    Handler.chain_port = args.remote_port

    server = ForwardServer(("127.0.0.1", args.local_port), Handler)
    print(f"[tunnel] 127.0.0.1:{args.local_port} -> "
          f"{HOST}:{PORT} -> {args.remote_host}:{args.remote_port}", flush=True)
    print("[tunnel] 保持运行；Ctrl+C 结束", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[tunnel] 已停止", flush=True)
    finally:
        server.shutdown()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
