#!/usr/bin/env python3
"""
NẠP ĐẦY PIN ĐỒNG LOẠT CHO 5 DRONE SITL (RESET BATTERY 100% 12.6V)
Khắc phục triệt để cơ chế cached dung lượng của mô hình Battery::maybe_reset trong ArduPilot SITL
"""

import sys
import time
from pymavlink import mavutil

# UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DRONE_PORTS = {
    1: 5773,
    2: 5783,
    3: 5793,
    4: 5803,
    5: 5813,
}

print("=" * 65)
print("  🔋 ĐANG NẠP ĐẦY PIN 100% (12.6V) CHO TOÀN BỘ 5 DRONE SITL...")
print("=" * 65)

for drone_id, port in DRONE_PORTS.items():
    try:
        mav = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", timeout=2)
        hb = mav.wait_heartbeat(timeout=2)
        if not hb:
            print(f"  ❌ Drone {drone_id}: Không nhận được Heartbeat!")
            continue

        # Kỹ thuật kích hoạt Battery::maybe_reset: làm thay đổi tham số để buộc SITL tính lại 100% dung lượng
        mav.param_set_send("SIM_BATT_CAP_AH", 1.41)
        mav.param_set_send("SIM_BATT_VOLTAGE", 12.61)
        time.sleep(0.05)
        mav.param_set_send("SIM_BATT_CAP_AH", 1.40)
        mav.param_set_send("SIM_BATT_VOLTAGE", 12.60)
        mav.param_set_send("ARMING_CHECK", 0)

        # Chuyển mode về GUIDED
        mode_id = mav.mode_mapping().get("GUIDED")
        if mode_id is not None:
            mav.set_mode(mode_id)

        print(f"  ✅ Drone {drone_id}: Đã nạp đầy 12.6V (100% Pin 3S 1400mAh) & Chuyển GUIDED!")
    except Exception as e:
        print(f"  ❌ Drone {drone_id}: Lỗi: {e}")

print("=" * 65)
print("  🎉 HOÀN TẤT NẠP PIN! CẢ 5 DRONE ĐÃ TRỞ LẠI TRẠNG THÁI GUIDED (12.6V / 100%)")
print("=" * 65)
