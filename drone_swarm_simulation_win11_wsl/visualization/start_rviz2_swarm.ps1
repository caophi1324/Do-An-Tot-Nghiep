# =================================================================
# RViz2 5-Drone Swarm 3D Viewer (Native Windows 10 PowerShell)
# =================================================================
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  STARTING RVIZ2 NATIVE WINDOWS 10 (100% DIRECT SITL TELEMETRY)" -ForegroundColor Yellow
Write-Host "  KHONG CAN WSL2 - KHONG CAN MAVROS - TOI UU 60 FPS" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$condaBat = "C:\ProgramData\miniforge3\condabin\conda.bat"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$rvizConfig = Join-Path $projectRoot "config\swarm_view.rviz"
$pubScript = Join-Path $scriptDir "drone_3d_mesh_publisher.py"

Write-Host "[1/2] Khoi dong 3D Mesh Publisher va mo RViz2..." -ForegroundColor Gray
$cmd = "call `"$condaBat`" activate ros2_win && python `"$pubScript`" 5"
$pubProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmd" -WindowStyle Hidden -PassThru

Write-Host "[2/2] Mo RViz2 Windows..." -ForegroundColor Gray
Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  [OK] RVIZ2 WINDOWS DANG CHAY!" -ForegroundColor Yellow
Write-Host "  Hay mo mot cua so PowerShell khac va chay lenh sau:" -ForegroundColor White
Write-Host "     python missions\show_30years_smooth.py" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""

$rvizCmd = "call `"$condaBat`" activate ros2_win && rviz2 -d `"$rvizConfig`""
cmd.exe /c $rvizCmd

Write-Host ""
Write-Host "[*] RViz2 da dong. Dang don dep tien trinh..." -ForegroundColor Yellow
if ($pubProcess -and -not $pubProcess.HasExited) {
    Stop-Process -Id $pubProcess.Id -Force -ErrorAction SilentlyContinue
}
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*Drone3DPublisher*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "[*] Don dep hoan tat!" -ForegroundColor Green
