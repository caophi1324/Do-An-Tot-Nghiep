@echo off
chcp 65001 > nul
title DRONE SWARM GCS - MISSION CONTROL (PYQT6)
echo =========================================================
echo   KHOI DONG DRONE SWARM GROUND CONTROL STATION (GCS)
echo   Nen tang: PyQt6 Native Desktop GUI
echo =========================================================
cd /d "%~dp0"
python app_gcs\main.py
if errorlevel 1 (
    echo.
    echo [LOI] Khong the khoi chay ung dung! Kiem tra thu vien PyQt6.
    pause
)
