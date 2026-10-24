@echo off
chcp 65001 > nul
title GAZEBO HARMONIC 3D SWARM VIEWER (WSL2)
echo =========================================================
echo   KHOI DONG GAZEBO HARMONIC 3D SWARM VISUALIZATION
echo   Moi truong: WSL2 (Ubuntu 24.04) + WSLg Do Hoa Windows 11
echo =========================================================
cd /d "%~dp0"

set COUNT=%1
if "%COUNT%"=="" set COUNT=3

echo [*] Kiem tra WSL2 va Gazebo Harmonic...
wsl gz sim --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay gz sim trong WSL2!
    pause
    exit /b 1
)

echo [*] Dang khoi chay Gazebo Harmonic Swarm (%COUNT% Drones) qua WSL2...
wsl bash -c "cd /mnt/d/TheProject/drone_swarm_simulation_win11_wsl && bash simulation/run_gazebo_swarm.sh %COUNT%"
pause
