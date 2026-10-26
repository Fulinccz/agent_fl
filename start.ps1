# FulinAI 一键启动（Windows PowerShell）
# 用法：右键"使用 PowerShell 运行"，或在仓库根目录执行：powershell -File .\start.ps1
# 启动内容：MySQL + Redis（Docker）→ Langfuse（独立 compose 项目，可选）→ AI Service(:8000) → Gateway(:8080) → Frontend(:5173)
# 停止用 .\stop.ps1；llama-server(:8091) 故意保留以便下次秒开（provider 会复用健康实例）

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Wait-Port([int]$Port, [int]$TimeoutSec, [string]$Name) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($c) { Write-Host "[OK] $Name 就绪 (:$Port)" -ForegroundColor Green; return $true }
        Start-Sleep -Seconds 2
    }
    Write-Host "[WARN] $Name ${TimeoutSec}s 内未就绪，请到对应窗口查看日志" -ForegroundColor Yellow
    return $false
}

# 1. 基础设施
Write-Host "==> 启动 MySQL / Redis (Docker)..." -ForegroundColor Cyan
docker compose -f "$root\docker-compose.yml" up -d mysql redis | Out-Null
foreach ($c in @("fulin-mysql", "fulin-redis")) {
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        $h = docker inspect -f "{{.State.Health.Status}}" $c 2>$null
        if ($h -eq "healthy") { Write-Host "[OK] $c healthy" -ForegroundColor Green; break }
        Start-Sleep -Seconds 3
    }
}

# 2. Langfuse 可观测性（独立 compose 项目，失败不影响主链路）
$lfStarted = $false
$lfCompose = Join-Path $root "langfuse\docker-compose.yml"
if (Test-Path $lfCompose) {
    if (Test-Path (Join-Path $root "langfuse\.env")) {
        Write-Host "==> 启动 Langfuse (Docker)..." -ForegroundColor Cyan
        # 注意：不能写 2>&1——docker 把进度打到 stderr，在 EAP=Stop 下会被当成异常抛出
        docker compose -f $lfCompose up -d | Out-Null
        if ($LASTEXITCODE -eq 0) { $lfStarted = $true }
        else { Write-Host "[WARN] Langfuse 启动失败（不影响主链路），稍后可单独排查" -ForegroundColor Yellow }
    } else {
        Write-Host "[WARN] 未找到 langfuse\.env，跳过 Langfuse（cp langfuse\.env.example langfuse\.env 后重试）" -ForegroundColor Yellow
    }
}

# 3. 三个应用服务，各占一个新窗口（日志各自可见，关窗口即停单个服务）
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "==> 启动 AI Service / Gateway / Frontend（各开新窗口）..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend\api-service'; `$Host.UI.RawUI.WindowTitle='FulinAI api-service'; & '$python' main.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend\api-gateway'; `$Host.UI.RawUI.WindowTitle='FulinAI gateway'; go run cmd/server/main.go"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; `$Host.UI.RawUI.WindowTitle='FulinAI frontend'; npm run dev"

# 4. 等待并就绪后开浏览器（本地模型预热约 30-60s，首条回复慢属正常）
Wait-Port 8000 90  "AI Service" | Out-Null
Wait-Port 8080 60  "Gateway"     | Out-Null
if (Wait-Port 5173 60 "Frontend") { Start-Process "http://localhost:5173" }

# 5. Langfuse 冷启动较慢（ClickHouse 迁移 + Next.js），放在开浏览器之后确认，不阻塞主界面
if ($lfStarted) { Wait-Port 3000 120 "Langfuse" | Out-Null }

Write-Host "`n全部拉起。停止：powershell -File .\stop.ps1" -ForegroundColor Cyan
