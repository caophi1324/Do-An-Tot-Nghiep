#!/usr/bin/env python3
"""
DRONEEXE - Single Drone SITL Launcher (Strict Real-World Safety Mode)
=============================================================================
Khởi chạy 1 instance SITL Drone 1 với file tham số an toàn nghiêm ngặt:
- Nạp config/sitl_hardware_real_safety.parm
- Model Micro Drone 240g (models/micro_drone_240g.json)
- Mở cổng SERIAL0 (5770 GCS), SERIAL1 (5772 MAVROS), SERIAL2 (5773 Python Mission)
=============================================================================
"""

import subprocess
import time
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thư mục chứa SITL của Mission Planner
sitl_dir = Path(os.environ['USERPROFILE']) / "Documents" / "Mission Planner" / "sitl"
arducopter_exe = sitl_dir / "ArduCopter.exe"
default_params = sitl_dir / "default_params" / "copter.parm"

if not arducopter_exe.exists():
    print(f"Error: Không tìm thấy file giả lập ở {arducopter_exe}")
    sys.exit(1)

# File tham số an toàn nghiêm ngặt riêng biệt
safety_parm = Path(__file__).resolve().parent.parent / "config" / "sitl_hardware_real_safety.parm"
if safety_parm.exists():
    defaults_arg = f"{default_params},{safety_parm}"
else:
    print(f"Warning: Không tìm thấy {safety_parm}, sử dụng copter.parm mặc định.")
    defaults_arg = str(default_params)

model_json = Path(__file__).resolve().parent.parent / "models" / "micro_drone_240g.json"
if model_json.exists():
    model_arg = f"+:{model_json.as_posix()}"
else:
    model_arg = "+"

print("=" * 65)
print("  🛡️ KHỞI CHẠY SITL DRONE 1: CHẾ ĐỘ MÔ PHỎNG AN TOÀN THỰC TẾ")
print("  Vật lý: Micro Drone 240g (Motor 1404, Pin 3S 1400mAh)")
print("  Cảm biến: GY-912 (10DOF) + MicoAir MTF-02 (Optical Flow + ToF)")
print("  Rào cản: ARMING_CHECK=1 | FS_GCS=1 | FENCE_ALT=2.5m | FENCE_RAD=8m")
print("  - SERIAL0 GCS    : tcp:127.0.0.1:5770")
print("  - SERIAL1 MAVROS : tcp:127.0.0.1:5772")
print("  - SERIAL2 Python : tcp:127.0.0.1:5773")
print("=" * 65)
print("Bấm Ctrl+C để tắt mô phỏng.\n")

# Chạy instance Drone 1 trong thư mục riêng
instance_dir = sitl_dir / "safety_test_1"
instance_dir.mkdir(parents=True, exist_ok=True)
eeprom_file = instance_dir / "eeprom.bin"
if eeprom_file.exists():
    try:
        eeprom_file.unlink()
        print("[*] Đã dọn dẹp eeprom.bin cũ để nạp mới toàn bộ tham số an toàn.")
    except Exception:
        pass

home_str = "-35.3632621,149.1652374,584,353"

cmd = [
    str(arducopter_exe),
    "-I", "1",
    "--sysid", "1",
    "--model", model_arg,
    "--speedup", "1",
    "--home", home_str,
    "--defaults", defaults_arg,
    "--serial0", "tcp:0",
    "--serial1", "tcp:2",
    "--serial2", "tcp:3",
]

process = None
try:
    process = subprocess.Popen(cmd, cwd=str(instance_dir))
    print(f"[*] Đã khởi động ArduCopter SITL PID {process.pid}. Đang chạy...")
    process.wait()
except KeyboardInterrupt:
    print("\n[!] Đang tắt tiến trình SITL...")
    if process:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
    print("[*] Đã tắt sạch sẽ.")
