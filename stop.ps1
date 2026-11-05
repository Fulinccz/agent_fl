﻿﻿﻿# FulinAI 一键停止：按端口定位进程（8000/8080/5173），并停止 MySQL/Redis 容器
# Langfuse 是可选观测栈（默认不随 start.ps1 启动），这里一并停掉以免残留占用内存
# llama-server(:8091) 不在停止列表——作为孤儿进程保留，下次启动 provider 直接复用（秒开）
$root = $PSScriptRoot

foreach ($port in @(5173, 8080, 8000)) {
    $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "停止 :$port -> $($proc.ProcessName) (PID $p)"
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
}

docker compose -f "$root\docker-compose.yml" stop mysql redis | Out-Null
Write-Host "MySQL / Redis 容器已停止。" -ForegroundColor Green

# Langfuse 是独立 compose 项目，根 compose 管不到，需用自己那份文件停止
$lfCompose = Join-Path $root "langfuse\docker-compose.yml"
if (Test-Path $lfCompose) {
    docker compose -f $lfCompose stop | Out-Null
    Write-Host "Langfuse 容器已停止（数据与配置保留，下次启动无需重建）。" -ForegroundColor Green
}

Write-Host "完成。" -ForegroundColor Green
