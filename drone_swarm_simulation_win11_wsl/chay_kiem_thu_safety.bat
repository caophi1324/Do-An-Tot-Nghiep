@echo off
chcp 65001 > nul
title KIEM THU AN TOAN DRONE THUC TE (STRICT SAFETY)
cls

echo =====================================================================
echo   🛡️ HỆ THỐNG MÔ PHỎNG AN TOÀN PHẦN CỨNG THỰC TẾ (SITL SAFETY HARNESS)
echo   Cấu hình: ARMING_CHECK=1 ^| Failsafe GCS ^| Geo-Fence 2.5m ^| Pin 3S
echo =====================================================================
echo.
echo   [1] Khởi chạy SITL Drone 1 (Chế độ An Toàn Nghiêm Ngặt)
echo   [2] Chạy Kịch bản Kiểm thử An toàn (Interactive Test Menu)
echo   [3] Chạy 1-Click: Bật SITL ngầm + Tự động Chạy Kiểm thử
echo   [4] Tắt toàn bộ tiến trình ArduCopter SITL
echo   [0] Thoát
echo.
echo =====================================================================
set /p opt="👉 Nhập lựa chọn của bạn (0-4): "

if "%opt%"=="1" (
    echo [*] Đang khởi chạy SITL Drone 1 với cấu hình sitl_hardware_real_safety.parm...
    ".venv\Scripts\python.exe" simulation\start_single_safety_sitl.py
    pause
    goto end
)

if "%opt%"=="2" (
    echo [*] Đang kết nối và chạy Kịch bản Kiểm thử An toàn...
    ".venv\Scripts\python.exe" missions\test_drone_real_safety.py
    pause
    goto end
)

if "%opt%"=="3" (
    echo [*] [1/3] Đang tắt các tiến trình SITL cũ...
    taskkill /F /IM ArduCopter.exe > nul 2>&1
    timeout /t 1 /nobreak > nul

    echo [*] [2/3] Đang bật SITL Drone 1 (Safety Mode) ngầm...
    start "" /B ".venv\Scripts\python.exe" simulation\start_single_safety_sitl.py

    echo [*] [3/3] Chờ EKF3 và cảm biến SITL khởi tạo (10 giây)...
    timeout /t 10 /nobreak > nul

    echo [*] Bắt đầu phiên kiểm thử an toàn...
    ".venv\Scripts\python.exe" missions\test_drone_real_safety.py

    echo.
    echo [*] Dọn dẹp tiến trình...
    taskkill /F /IM ArduCopter.exe > nul 2>&1
    pause
    goto end
)

if "%opt%"=="4" (
    echo [*] Đang tắt sạch các tiến trình ArduCopter.exe...
    taskkill /F /IM ArduCopter.exe > nul 2>&1
    echo [*] Đã tắt xong!
    pause
    goto end
)

:end
