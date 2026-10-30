# Langfuse 观测栈独立启停（默认不随 start.ps1 启动，避免常驻 2-3GB 内存）
# 用法：powershell -File .\scripts\langfuse.ps1 start|stop|status
# tracing 未配置/不可用时后端自动降级为 no-op，主链路（简历优化）不受影响
param(
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$compose = Join-Path $root "langfuse\docker-compose.yml"
$envFile = Join-Path $root "langfuse\.env"

if (-not (Test-Path $compose)) { throw "未找到 $compose" }

switch ($Action) {
    "start" {
        if (-not (Test-Path $envFile)) {
            throw "未找到 langfuse\.env：先复制 langfuse\.env.example 为 langfuse\.env 再启动"
        }
        Write-Host "==> 启动 Langfuse (Docker)..." -ForegroundColor Cyan
        # 注意：不要加 2>&1，docker 把进度打到 stderr，在 EAP=Stop 下会被当成异常抛出
        docker compose -f $compose up -d | Out-Null

        # ClickHouse 迁移 + Next.js 冷启动较慢，最多等 150s
        $deadline = (Get-Date).AddSeconds(150)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            if (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue) {
                $ready = $true
                break
            }
            Start-Sleep -Seconds 3
        }
        if ($ready) {
            Write-Host "[OK] Langfuse 就绪：http://localhost:3000" -ForegroundColor Green
            Start-Process "http://localhost:3000"
        } else {
            Write-Host "[WARN] 150s 内未就绪，查看日志：docker compose -f langfuse\docker-compose.yml logs -f" -ForegroundColor Yellow
        }
    }
    "stop" {
        docker compose -f $compose stop | Out-Null
        Write-Host "[OK] Langfuse 已停止（数据与配置保留，下次 start 无需重建）" -ForegroundColor Green
    }
    "status" {
        docker compose -f $compose ps
    }
}
