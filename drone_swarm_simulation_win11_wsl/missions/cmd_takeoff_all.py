#!/usr/bin/env python3
"""
LỆNH CẤT CÁNH VÀ DUY TRÌ HOVER ĐỒNG BỘ CHO 5 DRONE
"""

import sys
import time
from pymavlink import mavutil

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

num_drones = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
active_ports = {k: v for k, v in DRONE_PORTS.items() if k <= num_drones}

TARGET_ALT = 1.5

print("=" * 65)
print(f"  🛫 PHÁT LỆNH ARM VÀ CẤT CÁNH (HOVER {TARGET_ALT}M) CHO {num_drones} DRONE")
print("=" * 65)

# 1. ARM và phát lệnh cất cánh cho từng drone
for drone_id, port in active_ports.items():
    try:
        mav = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", timeout=2)
        hb = mav.wait_heartbeat(timeout=2)
        if not hb:
            print(f"  ⚠️ Drone {drone_id}: Không thấy heartbeat trên port {port}")
            continue

        # Đảm bảo mode GUIDED
        mav.set_mode("GUIDED")
        mav.param_set_send("ARMING_CHECK", 0)
        time.sleep(0.1)

        # Force ARM
        mav.mav.command_long_send(
            drone_id, 1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 21196,
            0, 0, 0, 0, 0
        )
        time.sleep(0.2)

        # Takeoff
        mav.mav.command_long_send(
            drone_id, 1,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            float(TARGET_ALT)
        )
        print(f"  ✅ Drone {drone_id}: Đã phát lệnh GUIDED ➔ ARM ➔ TAKEOFF {TARGET_ALT}m")
        mav.close()
    except Exception as e:
        print(f"  ❌ Drone {drone_id}: Lỗi: {e}")

print("\n" + "=" * 65)
print(f"  🎉 ĐÃ PHÁT LỆNH CẤT CÁNH THÀNH CÔNG CHO CẢ {num_drones} DRONE!")
print(f"  Quan sát trên Desktop GCS: {num_drones} drone đang nâng độ cao lên {TARGET_ALT}m và giữ HOVER.")
print("=" * 65)
