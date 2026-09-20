# 本地评测脚本模板（请复制为 run_eval_all.ps1 后使用）
#
# 为什么要有这个模板：
#   评测需要连本项目自己的 MySQL（docker-compose 映射在宿主机 3307），
#   需要数据库口令。口令**不入库**，所以真正的 run_eval_all.ps1 已被 .gitignore 排除。
#
# 用法：
#   1) Copy-Item .\finetune\run_eval_all.example.ps1 .\finetune\run_eval_all.ps1
#   2) 把下面的口令改成你自己的（或改成从环境变量读取）
#   3) & .\finetune\run_eval_all.ps1
#
# 前置条件：
#   * Docker 里的 MySQL 已启动（宿主机端口 3307，见 docker-compose.yaml）
#   * finetune\data\preds\ 下有 base.jsonl / lora_ep1.jsonl / lora_ep2.jsonl / lora_ep3.jsonl
#
# 注意：本文件若含中文注释，必须以 **UTF-8 BOM** 保存，
#       否则 Windows PowerShell 5.1 会按 ANSI 解码，中文变乱码导致语法错。

Set-Location -Path (Join-Path $PSScriptRoot '..')

# ---- 凭据：优先用已存在的环境变量，缺失时才用下面的默认值 ----
# 评测本身不调用 LLM，但项目配置加载要求 LLM_API_KEY 存在（给个占位即可）。
if (-not $env:LLM_API_KEY)    { $env:LLM_API_KEY = 'not-needed-for-eval' }
if (-not $env:LLM_MODEL_NAME) { $env:LLM_MODEL_NAME = 'deepseek-flash' }

if (-not $env:DB_DW_HOST)     { $env:DB_DW_HOST     = '127.0.0.1' }   # 不能用 localhost（会走 IPv6）
if (-not $env:DB_DW_PORT)     { $env:DB_DW_PORT     = '3307' }        # 3306 被别的容器占用
if (-not $env:DB_DW_USER)     { $env:DB_DW_USER     = 'appuser' }
if (-not $env:DB_DW_PASSWORD) { $env:DB_DW_PASSWORD = '<你的数据库口令>' }
if (-not $env:DB_DW_DATABASE) { $env:DB_DW_DATABASE = 'dw' }

if (-not $env:DB_META_HOST)     { $env:DB_META_HOST     = '127.0.0.1' }
if (-not $env:DB_META_PORT)     { $env:DB_META_PORT     = '3307' }
if (-not $env:DB_META_USER)     { $env:DB_META_USER     = 'appuser' }
if (-not $env:DB_META_PASSWORD) { $env:DB_META_PASSWORD = '<你的数据库口令>' }
if (-not $env:DB_META_DATABASE) { $env:DB_META_DATABASE = 'meta' }

$py      = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$predDir = Join-Path (Get-Location) 'finetune\data\preds'

# 预测文件 -> 显示名
$jobs = [ordered]@{
    'base.jsonl'     = 'A - Qwen3-8B 原版（未微调）'
    'lora_ep1.jsonl' = 'B1 - Qwen3-8B LoRA 1 epoch'
    'lora_ep2.jsonl' = 'B2 - Qwen3-8B LoRA 2 epoch'
    'lora_ep3.jsonl' = 'B3 - Qwen3-8B LoRA 3 epoch'
}

foreach ($p in $jobs.Keys) {
    $pred = Join-Path $predDir $p
    if (-not (Test-Path $pred)) { Write-Host "[skip] missing $p"; continue }

    $tag = [IO.Path]::GetFileNameWithoutExtension($p)
    $log = Join-Path $predDir "eval_$tag.log"
    $det = Join-Path $predDir "$tag.details.jsonl"

    Write-Host ("[eval] {0}" -f $jobs[$p]) -ForegroundColor Cyan
    # 只保留 stdout（正式报表）；stderr 是 validate_sql 的 EXPLAIN 调试输出，丢弃
    & $py 'finetune/eval_ex.py' --db mysql --predictions $pred `
        --name $jobs[$p] --details-out $det 2>$null |
        Out-File -FilePath $log -Encoding utf8
    Write-Host ("       exit={0}" -f $LASTEXITCODE) -ForegroundColor Gray
}

Write-Host ""
Write-Host "===== SUMMARY =====" -ForegroundColor Green
foreach ($p in $jobs.Keys) {
    $tag = [IO.Path]::GetFileNameWithoutExtension($p)
    $log = Join-Path $predDir "eval_$tag.log"
    if (-not (Test-Path $log)) { continue }
    Write-Host "---- $tag ----"
    Get-Content $log -Encoding UTF8 |
        Select-String -Pattern '^(评测对象|测试集规模|L1 |L2 |L3 )' |
        ForEach-Object { Write-Host ("   " + $_.Line.Trim()) }
    Write-Host ""
}
