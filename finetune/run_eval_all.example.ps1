# 本地评测入口：用真实 MySQL 跑预测文件，算出三层指标
#
# 用法（在 data-agent 目录下）：
#     & .\finetune\run_eval_all.ps1
#     & .\finetune\run_eval_all.ps1 -Interim preds_v2      # 换一个预测目录
#
# 前置条件：
#   * docker 里的 MySQL 已启动（宿主机端口 3307，见 docker-compose.yaml）
#   * finetune\data\<Interim>\ 下有 base.jsonl / lora_ep1.jsonl / ...
#
# 说明：评测走项目自带配置（conf/app_config.yaml），凭据一律用环境变量注入，
#       不写死口令到脚本里；--db mysql 因为 MySQL 方言才有参考价值。
#       本文件必须以 UTF-8 BOM 保存，否则 Windows PowerShell 5.1 会把中文读成乱码而报语法错。

param(
    # 预测文件所在子目录（相对 finetune\data\），默认 preds
    [string]$Interim = "preds"
)

Set-Location -Path (Join-Path $PSScriptRoot '..')

# ---- 凭据（与 start.ps1 保持一致；评测不需要真实 LLM Key，但配置加载要求它存在）----
if (-not $env:LLM_API_KEY)    { $env:LLM_API_KEY = 'not-needed-for-eval' }
if (-not $env:LLM_MODEL_NAME) { $env:LLM_MODEL_NAME = 'deepseek-flash' }
$env:DB_DW_HOST       = '127.0.0.1'
$env:DB_DW_PORT       = '3307'
$env:DB_DW_USER       = 'appuser'
if (-not $env:DB_DW_PASSWORD) { $env:DB_DW_PASSWORD = '<你的数据库口令>' }
$env:DB_DW_DATABASE   = 'dw'
$env:DB_META_HOST     = '127.0.0.1'
$env:DB_META_PORT     = '3307'
$env:DB_META_USER     = 'appuser'
if (-not $env:DB_META_PASSWORD) { $env:DB_META_PASSWORD = '<你的数据库口令>' }
$env:DB_META_DATABASE = 'meta'

$py      = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$predDir = Join-Path (Get-Location) "finetune\data\$Interim"

if (-not (Test-Path $predDir)) {
    Write-Host "[错误] 找不到预测目录 $predDir" -ForegroundColor Red
    exit 2
}

# 预测文件 -> 显示名
$jobs = [ordered]@{
    'base.jsonl'     = 'A - Qwen3-8B 原版（未微调）'
    'lora_ep1.jsonl' = 'B1 - LoRA 1 epoch'
    'lora_ep2.jsonl' = 'B2 - LoRA 2 epoch'
    'lora_ep3.jsonl' = 'B3 - LoRA 3 epoch'
}

Write-Host "预测目录：$predDir" -ForegroundColor DarkGray
$ran = 0
foreach ($p in $jobs.Keys) {
    $pred = Join-Path $predDir $p
    if (-not (Test-Path $pred)) { Write-Host "[skip] 缺少 $p" -ForegroundColor Yellow; continue }

    $tag = [IO.Path]::GetFileNameWithoutExtension($p)
    $log = Join-Path $predDir "eval_$tag.log"
    $det = Join-Path $predDir "$tag.details.jsonl"

    Write-Host ("[eval] {0}" -f $jobs[$p]) -ForegroundColor Cyan
    # 只保留 stdout（正式报表）；stderr 是 validate_sql 的 EXPLAIN 调试输出，丢弃
    & $py 'finetune/eval_ex.py' --db mysql --predictions $pred `
        --name $jobs[$p] --details-out $det 2>$null |
        Out-File -FilePath $log -Encoding utf8
    Write-Host ("       exit={0}" -f $LASTEXITCODE) -ForegroundColor Gray
    $ran++
}
if ($ran -eq 0) { Write-Host "[错误] 该目录下没有任何预测文件" -ForegroundColor Red; exit 2 }

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
