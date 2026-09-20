# 掌柜问数 · 本地启动（走微调后的 Qwen3-8B）
#
# 用法（在 data-agent 目录下）：
#     .\start_qwen.ps1
#
# 它会用**微调模型**而不是云端 deepseek 启动后端。与 start.ps1 的区别只有 LLM 那几项。
#
# 前置条件（缺一个都起不来）：
#   1. 租卡机器上模型服务已启动：
#        cd /root/autodl-tmp
#        PYTHONUNBUFFERED=1 nohup python serve_qwen.py \
#            --adapter /root/autodl-tmp/output/qwen3-8b-lora-v2-1ep-ep1 --port 8000 \
#            > logs/serve_qwen.log 2>&1 &
#   2. 隧道已建好（另开一个窗口，保持运行）：
#        $env:DSH_SSH_PASSWORD='<租卡机器密码>'; .\.venv\Scripts\python.exe finetune\ssh_tunnel.py
#   3. 本机 docker 的 MySQL/Qdrant/ES/TEI 都在跑（3307/6333/9200/8081）
#
# 关于端口：8000 被 WSL/Docker 占、8001 被别的项目占，所以本项目用 8010。
#
# 本文件必须以 UTF-8 BOM 保存，否则 Windows PowerShell 5.1 会把中文读成乱码而报语法错。

Set-Location -Path $PSScriptRoot

# ============ LLM：指向本机隧道 -> 租卡机器上的微调模型 ============
$env:LLM_PROVIDER   = 'openai_compatible'
$env:LLM_BASE_URL   = 'http://127.0.0.1:8000/v1'
$env:LLM_MODEL_NAME = 'qwen3-8b-lora'
# 本地端点不校验 key；给占位值即可（配置里空值会被 llm.py 兜底成 EMPTY）
$env:LLM_API_KEY    = 'EMPTY'

# ============ MySQL：host 必须 127.0.0.1（localhost 会走 IPv6 连不上）============
$env:DB_META_HOST     = '127.0.0.1'
$env:DB_DW_HOST       = '127.0.0.1'
$env:DB_META_PORT     = '3307'   # 3306 被其它容器占用
$env:DB_DW_PORT       = '3307'
$env:DB_META_USER     = 'appuser'
$env:DB_DW_USER       = 'appuser'
$env:DB_META_PASSWORD = 'AppPwd123'
$env:DB_DW_PASSWORD   = 'AppPwd123'
$env:DB_META_DATABASE = 'meta'
$env:DB_DW_DATABASE   = 'dw'

# ============ 服务端口 ============
$env:APP_PORT = '8010'

Write-Host "启动掌柜问数（微调模型版）..." -ForegroundColor Cyan
Write-Host ("  LLM   : {0}  @ {1}" -f $env:LLM_MODEL_NAME, $env:LLM_BASE_URL) -ForegroundColor Gray
Write-Host ("  MySQL : {0}:{1}/{2}" -f $env:DB_DW_HOST, $env:DB_DW_PORT, $env:DB_DW_DATABASE) -ForegroundColor Gray
Write-Host ("  接口  : http://127.0.0.1:{0}/api/query" -f $env:APP_PORT) -ForegroundColor Gray
Write-Host ""

# 先探一下模型服务，避免"启动了但一调用就超时"
try {
    $r = Invoke-WebRequest -Uri "$($env:LLM_BASE_URL)/models" -TimeoutSec 8 -UseBasicParsing
    Write-Host ("  ✅ 模型服务可达：{0}" -f $r.Content) -ForegroundColor Green
} catch {
    Write-Host "  ❌ 模型服务不可达 —— 检查租卡机器上的 serve_qwen.py 和隧道" -ForegroundColor Red
    Write-Host ("     {0}" -f $_.Exception.Message) -ForegroundColor DarkGray
    Write-Host "     仍然继续启动后端，但问数请求会失败。" -ForegroundColor DarkGray
}
Write-Host ""

& "$PSScriptRoot\.venv\Scripts\python.exe" -m main
