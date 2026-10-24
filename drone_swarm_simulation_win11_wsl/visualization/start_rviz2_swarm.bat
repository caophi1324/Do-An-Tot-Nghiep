@echo off
title RViz2 5-Drone Swarm 3D Viewer (Native Windows 10)
echo =================================================================
echo   STARTING RVIZ2 NATIVE WINDOWS 10 (100%% DIRECT SITL TELEMETRY)
echo   KHONG CAN WSL2 - KHONG CAN MAVROS - TOI UU 60 FPS
echo =================================================================
echo.

echo [1/2] Nap moi truong ROS 2 Windows (ros2_win)...
call C:\ProgramData\miniforge3\condabin\conda.bat activate ros2_win
if errorlevel 1 (
    echo [ERROR] Khong the kich hoat conda env ros2_win!
    pause
    exit /b 1
)

echo [2/2] Khoi dong 3D Mesh Publisher va mo RViz2 Windows...
taskkill /F /FI "WINDOWTITLE eq Drone3DPublisher*" /IM python.exe >nul 2>&1
start "Drone3DPublisher" /B python "%~dp0drone_3d_mesh_publisher.py" 5

echo.
echo =================================================================
echo  [OK] RVIZ2 WINDOWS DANG CHAY!
echo  Hay mo mot cua so PowerShell khac va chay:
echo     python missions\show_30years_smooth.py
echo =================================================================
echo.

rviz2 -d "%~dp0..\config\swarm_view.rviz"

echo.
echo [*] RViz2 da dong. Dang don dep tien trinh ngam...
taskkill /F /FI "WINDOWTITLE eq Drone3DPublisher*" /IM python.exe >nul 2>&1
echo [*] Don dep hoan tat!
