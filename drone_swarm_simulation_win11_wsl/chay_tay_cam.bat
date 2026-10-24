@echo off
chcp 65001 >nul
title TRÌNH ĐIỀU KHIỂN TAY CẦM DRONE (GAMEPAD PILOT)

echo =====================================================================
echo          🎮 TRÌNH ĐIỀU KHIỂN TAY CẦM DRONE (GAMEPAD / RC)
echo   Hỗ trợ: Xbox, PS4/PS5, Logitech, Flysky/Radiomaster, Generic USB
echo =====================================================================
echo.
echo  [1] Đo biên độ gạt tối đa & Hiệu chuẩn (Min/Max Range Calibration)
echo  [2] Kiểm tra phần cứng live (Xem Cần gạt, Nút bấm, D-Pad)
echo  [3] 🚀 Lái Drone 1 Dẫn Đàn + Bầy Đàn Bám Đuôi & Tự Né APF (All-in-One Swarm Leader)
echo  [4] Điều khiển 1 Drone Độc Lập (Drone 1 Solo)
echo  [5] Điều khiển Toàn Bầy 5 Drone (Swarm Pilot - Đổi drone bằng LB/RB)
echo  [6] Thoát
echo.
echo =====================================================================
set /p choice="👉 Nhập lựa chọn của bạn (1/2/3/4/5/6): "

if "%choice%"=="1" (
    cls
    python scripts\calibrate_gamepad.py
    pause
    goto end
)

if "%choice%"=="2" (
    cls
    python scripts\test_gamepad_diagnostics.py
    pause
    goto end
)

if "%choice%"=="3" (
    cls
    python missions\gamepad_leader_apf_swarm.py
    pause
    goto end
)

if "%choice%"=="4" (
    cls
    python missions\gamepad_single_drone.py
    pause
    goto end
)

if "%choice%"=="5" (
    cls
    python missions\gamepad_swarm_controller.py
    pause
    goto end
)

:end
