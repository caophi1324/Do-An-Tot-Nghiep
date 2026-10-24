@echo off
chcp 65001 > nul
title ORCA RVO2 SWARM GCS - MISSION CONTROL (PYQT6)
echo =========================================================
echo   ORCA / RVO2 SWARM GROUND CONTROL STATION
echo   Thuat toan: ORCA/RVO2 + Hungarian + Minimum Snap
echo   Nen tang: PyQt6 Native Desktop GUI
echo =========================================================
cd /d "%~dp0"
python -m orca_swarm.app_gcs.main
if errorlevel 1 (
    echo.
    echo [LOI] Khong the khoi chay ung dung! Kiem tra thu vien PyQt6.
    pause
)
