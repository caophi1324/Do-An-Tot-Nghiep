@echo off
echo ======================================================
echo  [INFO] Dang tat toan bo tien trinh Swarm SITL va RViz2...
echo ======================================================

rem 1. Tat tien trinh trong WSL (MAVROS, SITL Linux, ROS2)
wsl bash -c "killall -9 arducopter ArduCopter mavproxy.py sim_vehicle.py mavros_node drone_3d_mesh_publisher.py gz 'gz sim' 2>/dev/null || true" >nul 2>&1

rem 2. Tat RViz2 va ArduCopter tren Windows
taskkill /F /IM rviz2.exe >nul 2>&1
taskkill /F /IM ArduCopter.exe >nul 2>&1
powershell -NoProfile -Command "Get-Process -Name ArduCopter,rviz2 -ErrorAction SilentlyContinue | Stop-Process -Force" >nul 2>&1

rem 3. Tat cac tien trinh Python lien quan den Drone Swarm
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*drone_3d_mesh_publisher*' -or $_.CommandLine -like '*show_30years*' -or $_.CommandLine -like '*start_swarm_sitl*' -or $_.CommandLine -like '*monitor_swarm*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

echo [OK] DA TAT SACH SE TOAN BO TIEN TRINH!
echo.
pause
