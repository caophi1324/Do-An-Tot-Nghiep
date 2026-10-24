# ======================================================
# Kill All Drone Swarm Processes (Windows & WSL2)
# ======================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " [INFO] Dang tat toan bo tien trinh Swarm SITL & RViz2..." -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Tat tien trinh trong WSL (MAVROS, SITL Linux, ROS2)
wsl bash -c "killall -9 arducopter ArduCopter mavproxy.py sim_vehicle.py mavros_node drone_3d_mesh_publisher.py gz 'gz sim' 2>/dev/null || true" 2>$null

# 2. Tat RViz2 va ArduCopter tren Windows
Get-Process -Name ArduCopter,rviz2 -ErrorAction SilentlyContinue | Stop-Process -Force

# 3. Tat cac tien trinh Python lien quan den Drone Swarm
Get-CimInstance Win32_Process | Where-Object { 
    $_.Name -eq 'python.exe' -and (
        $_.CommandLine -like '*drone_3d_mesh_publisher*' -or 
        $_.CommandLine -like '*show_30years*' -or 
        $_.CommandLine -like '*start_swarm_sitl*' -or 
        $_.CommandLine -like '*monitor_swarm*'
    ) 
} | ForEach-Object { 
    Stop-Process -Id $_.ProcessId -Force 
}

Write-Host "`n[OK] DA TAT SACH SE TOAN BO TIEN TRINH!" -ForegroundColor Green
