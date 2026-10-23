# FulinAI 一键停止：按端口定位进程（8000/8080/5173），并停止 MySQL/Redis 容器
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
Write-Host "MySQL / Redis 容器已停止。完成。" -ForegroundColor Green
