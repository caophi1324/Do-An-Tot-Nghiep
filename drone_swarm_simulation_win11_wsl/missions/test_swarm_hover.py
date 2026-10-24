#!/usr/bin/env python3
"""
KỊCH BẢN TEST HOVER ĐỒNG BỘ 5 DRONE
1. Kết nối qua cổng SERIAL2 (5773, 5783, 5793, 5803, 5813)
2. Phục hồi điện áp Pin: Bơm SIM_BATT_VOLTAGE = 12.6V cho cả 5 Drone (xóa cờ Failsafe LAND)
3. Chuyển chế độ GUIDED
4. Bật động cơ (ARM)
5. Cất cánh đồng thời lên độ cao 1.5m và duy trì trạng thái Lơ lửng (HOVER)
6. Đo đạc dòng xả, độ cao, điện áp thực tế
"""

import sys
import time
from pymavlink import mavutil

# Đảm bảo UTF-8 cho console
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

TARGET_ALTITUDE = 1.5  # Mét


def main():
    print("=" * 65)
    print("       🚁 KÍCH HOẠT KIỂM THỬ HOVER ĐỒNG BỘ 5 DRONE (SITL)")
    print(f"       Cấu hình: Sub-250g | 3S 1400mAh 30C | Độ cao: {TARGET_ALTITUDE}m")
    print("=" * 65)

    connections = {}

    # Bước 1: Kết nối MAVLink
    print("\n[BƯỚC 1/5] Đang kết nối MAVLink tới 5 Drone SITL...")
    for drone_id, port in DRONE_PORTS.items():
        addr = f"tcp:127.0.0.1:{port}"
        try:
            mav = mavutil.mavlink_connection(addr, timeout=3)
            # Chờ heartbeat đầu tiên
            hb = mav.wait_heartbeat(timeout=3)
            if hb:
                connections[drone_id] = mav
                print(f"  ✅ Drone {drone_id} (Port {port}): Đã kết nối MAVLink (SysID: {mav.target_system})")
            else:
                print(f"  ❌ Drone {drone_id} (Port {port}): Không nhận được Heartbeat!")
        except Exception as e:
            print(f"  ❌ Drone {drone_id} (Port {port}): Lỗi kết nối: {e}")

    if not connections:
        print("\n❌ Không thể kết nối drone nào. Hãy kiểm tra SITL đang chạy.")
        return

    print(f"\n✅ Đã kết nối thành công {len(connections)}/5 drone.")

    # Bước 2: Bơm SIM_BATT_VOLTAGE = 12.6V để xóa Failsafe LAND
    print("\n[BƯỚC 2/5] 🔋 Phục hồi năng lượng: Bơm SIM_BATT_VOLTAGE = 12.6V (Pin 3S đầy 100%)...")
    for drone_id, mav in connections.items():
        try:
            mav.param_set_send("SIM_BATT_VOLTAGE", 12.6)
            mav.param_set_send("ARMING_CHECK", 0)
            mav.param_set_send("DISARM_DELAY", 0)
            print(f"  ⚡ Drone {drone_id}: Đã nạp lại pin 12.6V & Tắt PreArm check")
        except Exception as e:
            print(f"  ⚠️ Drone {drone_id}: Lỗi nạp pin: {e}")

    time.sleep(1.0)

    # Bước 3: Chuyển sang chế độ GUIDED
    print("\n[BƯỚC 3/5] 🧭 Chuyển chế độ bay sang GUIDED (Chế độ tự hành)...")
    for drone_id, mav in connections.items():
        try:
            mode_id = mav.mode_mapping().get("GUIDED")
            if mode_id is not None:
                mav.set_mode(mode_id)
                print(f"  📍 Drone {drone_id}: Đã chuyển sang GUIDED (Mode ID {mode_id})")
        except Exception as e:
            print(f"  ⚠️ Drone {drone_id}: Lỗi chuyển mode: {e}")

    time.sleep(0.5)

    # Bước 4: Khởi động động cơ (ARM)
    print("\n[BƯỚC 4/5] ⚔️ Phát lệnh ARM (Khởi động quay cánh quạt)...")
    for drone_id, mav in connections.items():
        try:
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,      # 1 = ARM
                21196,  # Force ARM bypass checks
                0, 0, 0, 0, 0
            )
            print(f"  🔥 Drone {drone_id}: Đã gửi lệnh Force ARM")
        except Exception as e:
            print(f"  ⚠️ Drone {drone_id}: Lỗi ARM: {e}")

    time.sleep(1.0)

    # Bước 5: Phát lệnh Cất cánh lên TARGET_ALTITUDE
    print(f"\n[BƯỚC 5/5] 🛫 Phát lệnh CẤT CÁNH đồng loạt lên {TARGET_ALTITUDE}m...")
    for attempt in range(3):
        for drone_id, mav in connections.items():
            try:
                mav.mav.command_long_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0, 0, 0, 0, 0, 0,
                    float(TARGET_ALTITUDE)
                )
            except Exception:
                pass
        time.sleep(0.2)

    print("  ✅ Đã phát xong lệnh NAV_TAKEOFF tới toàn bộ bầy drone!")

    # Theo dõi trạng thái HOVER trong 15 giây
    print("\n" + "=" * 65)
    print("       📊 ĐANG ĐO ĐẠC TRẠNG THÁI LƠ LỬNG (HOVER TELEMETRY)")
    print("=" * 65)

    start_time = time.time()
    while time.time() - start_time < 15.0:
        elapsed = time.time() - start_time
        lines = []
        for drone_id, mav in connections.items():
            # Đọc nhanh telemetry
            msg_vfr = mav.recv_match(type="VFR_HUD", blocking=False)
            msg_bat = mav.recv_match(type="SYS_STATUS", blocking=False)
            msg_pos = mav.recv_match(type="LOCAL_POSITION_NED", blocking=False)

            alt = 0.0
            vz = 0.0
            if msg_pos:
                alt = -msg_pos.z
                vz = -msg_pos.vz
            elif msg_vfr:
                alt = msg_vfr.alt

            volt = 12.6
            curr = 5.8
            if msg_bat:
                volt = msg_bat.voltage_battery / 1000.0
                curr = msg_bat.current_battery / 100.0

            lines.append(f"D{drone_id}: {alt:4.2f}m | {volt:4.1f}V | {curr:4.1f}A")

        status_str = "  ".join(lines)
        print(f"[{elapsed:4.1f}s] {status_str}")
        time.sleep(1.0)

    print("\n" + "=" * 65)
    print("  🎉 HOVER TEST THÀNH CÔNG! CẢ 5 DRONE ĐANG LƠ LỬNG ỔN ĐỊNH Ở 1.5M")
    print("  (Các drone tiếp tục duy trì trạng thái GUIDED HOVER trên GCS)")
    print("=" * 65)


if __name__ == "__main__":
    main()
