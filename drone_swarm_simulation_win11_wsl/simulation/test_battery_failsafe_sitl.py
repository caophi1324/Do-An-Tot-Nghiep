#!/usr/bin/env python3
"""
KIỂM THỬ TỰ ĐỘNG CƠ CHẾ AN TOÀN PIN (BATTERY FAILSAFE) TRÊN ARDUPILOT SITL
Kịch bản:
1. Khởi động 1 Drone SITL (Drone 1) với file thông số sitl_hardware_gy912_mtf02.parm
2. Kết nối MAVLink, chuyển chế độ GUIDED, cất cánh lên 1.5m
3. Ổn định lơ lửng ở 1.5m với pin chuẩn 12.6V
4. Bắn sự cố sụt áp giả lập (Inject Low Battery): SIM_BATT_VOLTAGE = 10.2V (< 10.5V BATT_LOW_VOLT)
5. Lắng nghe và đo đạc:
   - ArduPilot phát hiện điện áp < 10.5V trong 3 giây
   - Kích hoạt BATT_FS_LOW_ACT = 1 (Tự động chuyển sang chế độ LAND)
   - Drone từ từ hạ cánh thẳng đứng từ 1.5m xuống 0.0m nhờ ToF
   - Khi chạm đất, động cơ tự động ngắt (DISARM) an toàn
6. Khôi phục điện áp 12.6V và dọn dẹp tiến trình SITL
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pymavlink import mavutil

# Đường dẫn SITL
sitl_dir = Path(os.environ['USERPROFILE']) / "Documents" / "Mission Planner" / "sitl"
arducopter_exe = sitl_dir / "ArduCopter.exe"
default_params = sitl_dir / "default_params" / "copter.parm"

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
hardware_parm = project_root / "config" / "sitl_hardware_gy912_mtf02.parm"

# Tên chế độ bay ArduCopter
MODE_NAMES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    17: "BRAKE",
}


def run_battery_failsafe_test():
    print("=" * 70)
    print("  KIỂM THỬ CƠ CHẾ AN TOÀN PIN (BATTERY FAILSAFE) TRÊN SITL")
    print("  Cấu hình: Sub-250g | Motor 1404 | Pin 3S 2200mAh | BATT_FS_LOW_ACT = LAND")
    print("=" * 70)

    # 1. Kiểm tra / Khởi động tiến trình SITL Drone 1
    tcp_port = 5773  # SERIAL2 port
    drone_dir = sitl_dir / "1"
    drone_dir.mkdir(parents=True, exist_ok=True)

    defaults_arg = f"{default_params},{hardware_parm}" if hardware_parm.exists() else str(default_params)

    cmd = [
        str(arducopter_exe),
        "-I", "1",
        "--sysid", "1",
        "--model", "+",
        "--speedup", "1",
        "--defaults", defaults_arg,
        "--home", "-35.3632621,149.1652374,584,353",
        "--serial0", "tcp:0",
        "--serial1", "tcp:2",
        "--serial2", "tcp:3",
    ]

    print(f"[*] Khởi động ArduCopter SITL Drone 1...")
    sitl_proc = subprocess.Popen(cmd, cwd=str(drone_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

    try:
        # 2. Kết nối MAVLink
        conn_str = f"tcp:127.0.0.1:{tcp_port}"
        print(f"[*] Đang kết nối MAVLink tới {conn_str}...")
        mav = None
        for attempt in range(15):
            try:
                mav = mavutil.mavlink_connection(conn_str, timeout=3.0)
                hb = mav.wait_heartbeat(timeout=3.0)
                if hb:
                    print(f"    [+] Đã nhận Heartbeat từ SYSID {hb.get_srcSystem()} (Type: {hb.type})")
                    break
            except Exception:
                pass
            time.sleep(1.0)

        if not mav or not hb:
            raise RuntimeError("Không thể kết nối MAVLink tới SITL sau 15 giây!")

        # Yêu cầu stream dữ liệu
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )

        # 3. Đồng bộ và xác nhận các tham số Failsafe
        print("\n[*] Kiểm tra tham số Battery Failsafe trong SITL...")
        params_to_set = {
            "BATT_MONITOR": 4,
            "BATT_LOW_VOLT": 10.5,
            "BATT_CRT_VOLT": 9.9,
            "BATT_FS_LOW_ACT": 1,    # 1 = LAND
            "BATT_FS_CRT_ACT": 1,    # 1 = LAND
            "BATT_LOW_TIMER": 3,     # 3 giây kích hoạt
            "SIM_BATT_VOLTAGE": 12.6 # Ban đầu pin đầy
        }

        for p_name, p_val in params_to_set.items():
            mav.mav.param_set_send(
                mav.target_system, mav.target_component,
                p_name.encode('ascii'), float(p_val),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            time.sleep(0.05)
        print("    [+] Đã nạp bộ tham số Failsafe chuẩn: BATT_LOW_VOLT=10.5V, BATT_FS_LOW_ACT=1 (LAND)")

        time.sleep(1.0)

        # 4. Chờ GPS & EKF3 lấy vị trí Home 3D
        print("\n[*] Đang chờ GPS 3D Fix & EKF3 thiết lập toạ độ Home (khoảng 15-25s)...")
        gps_ready = False
        start_wait = time.time()

        while time.time() - start_wait < 40.0:
            msg = mav.recv_match(type=['GPS_RAW_INT', 'STATUSTEXT', 'HEARTBEAT'], blocking=True, timeout=1.0)
            if msg:
                if msg.get_type() == 'GPS_RAW_INT':
                    if msg.fix_type >= 3:
                        gps_ready = True
                elif msg.get_type() == 'STATUSTEXT':
                    text = msg.text
                    print(f"    [SITL]: {text}")
                    if "Home" in text or "using GPS" in text or "origin set" in text:
                        gps_ready = True
                        break
            if gps_ready:
                break
            time.sleep(0.5)

        print(f"    [+] GPS & EKF3 đã sẵn sàng sau {time.time() - start_wait:.1f}s!")
        time.sleep(2.0)

        # 5. Chuyển chế độ GUIDED và ARM động cơ
        print("\n[*] Chuyển chế độ GUIDED và ARM động cơ Drone 1...")
        is_armed = False

        for arm_attempt in range(25):
            mav.mav.set_mode_send(
                mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4  # GUIDED
            )
            mav.mav.param_set_send(
                mav.target_system, mav.target_component,
                b"ARMING_CHECK", 0.0,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,      # 1 = ARM
                21196,  # Force ARM
                0, 0, 0, 0, 0
            )
            time.sleep(0.5)

            # Đọc tất cả tin nhắn từ ArduPilot để chẩn đoán
            while True:
                msg = mav.recv_match(type=['STATUSTEXT', 'COMMAND_ACK', 'HEARTBEAT'], blocking=False)
                if not msg:
                    break
                if msg.get_type() == 'STATUSTEXT':
                    print(f"    [SITL Text]: {msg.text}")
                elif msg.get_type() == 'COMMAND_ACK':
                    print(f"    [SITL Command ACK]: cmd={msg.command}, result={msg.result}")
                elif msg.get_type() == 'HEARTBEAT':
                    if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        is_armed = True
                        break

            if is_armed:
                print(f"    [+] ĐÃ ARM THÀNH CÔNG (sau {arm_attempt+1} lần thử)!")
                break

        if not is_armed:
            raise RuntimeError("Drone 1 không thể ARM sau 25 lần thử! Kiểm tra lại EKF/SITL.")

        # 5. Phát lệnh Cất Cánh (Takeoff 1.5m)
        target_alt = 1.5
        print(f"[*] Phát lệnh cất cánh GUIDED TAKEOFF lên {target_alt}m...")
        for _ in range(3):
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0, 0, 0,
                target_alt
            )
            time.sleep(0.3)

        # Chờ lên độ cao ổn định
        print("    [+] Đang theo dõi cao độ cất cánh...")
        climb_start = time.time()
        current_alt = 0.0
        while time.time() - climb_start < 15.0:
            msg = mav.recv_match(type=['GLOBAL_POSITION_INT', 'VFR_HUD', 'LOCAL_POSITION_NED'], blocking=True, timeout=1.0)
            if msg:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    current_alt = msg.relative_alt / 1000.0
                elif msg.get_type() == 'VFR_HUD':
                    current_alt = msg.alt
                elif msg.get_type() == 'LOCAL_POSITION_NED':
                    current_alt = -msg.z

            print(f"    -> Cao độ hiện tại: {current_alt:.2f} m", end="\r")
            if current_alt >= target_alt * 0.8:
                print(f"\n    [OK] Đã đạt độ cao lơ lửng ổn định: {current_alt:.2f}m")
                break
            time.sleep(0.3)

        if current_alt < 0.5:
            print("\n    [CẢNH BÁO] Drone chưa nhấc cánh, thử lại lệnh TAKEOFF...")
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, target_alt
            )
            time.sleep(3.0)

        time.sleep(2.0)

        # 7. KÍCH HOẠT SỰ CỐ PIN YẾU (INJECT LOW BATTERY)
        fault_volt = 10.2  # Dưới ngưỡng 10.5V
        print("\n" + "=" * 70)
        print(f"  ⚡ KÍCH HOẠT SỰ CỐ: Giả lập pin sụt áp xuống {fault_volt}V (Ngưỡng cảnh báo: 10.5V)!")
        print("=" * 70)

        mav.mav.param_set_send(
            mav.target_system, mav.target_component,
            b"SIM_BATT_VOLTAGE", float(fault_volt),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )

        # 8. Theo dõi phản ứng Failsafe
        print("\n[*] Lắng nghe và đo đạc phản ứng Failsafe từng giây...")
        print(f"{'Thời gian':<10} | {'Điện áp (V)':<12} | {'Chế độ bay':<14} | {'Độ cao Z (m)':<14} | {'Trạng thái'}")
        print("-" * 70)

        test_start = time.time()
        failsafe_triggered = False
        landed_and_disarmed = False

        volt = fault_volt
        mode_str = "GUIDED"
        alt = current_alt
        armed_str = "ARMED"

        while time.time() - test_start < 30.0:
            elapsed = time.time() - test_start

            # Đọc cạn các message mới nhất trong buffer
            while True:
                m = mav.recv_match(blocking=False)
                if not m:
                    break
                m_type = m.get_type()
                if m_type == 'HEARTBEAT':
                    custom_mode = m.custom_mode
                    mode_str = MODE_NAMES.get(custom_mode, f"MODE_{custom_mode}")
                    is_armed = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    armed_str = "ARMED" if is_armed else "DISARMED"
                elif m_type == 'SYS_STATUS':
                    if m.voltage_battery > 0:
                        volt = m.voltage_battery / 1000.0
                elif m_type == 'GLOBAL_POSITION_INT':
                    alt = m.relative_alt / 1000.0
                elif m_type == 'LOCAL_POSITION_NED':
                    alt = -m.z
                elif m_type == 'STATUSTEXT':
                    text = m.text
                    if any(k in text for k in ["Battery", "Failsafe", "Low", "Land", "Disarming"]):
                        print(f"    >>> [CẢNH BÁO ARDUPILOT]: {text}")

            print(f"{elapsed:6.1f}s     | {volt:6.2f} V      | {mode_str:<14} | {alt:6.2f} m        | {armed_str}")

            if mode_str == "LAND":
                if not failsafe_triggered:
                    failsafe_triggered = True
                    print(f"\n    [THÀNH CÔNG] ĐÃ KÍCH HOẠT BATTERY FAILSAFE! ArduPilot tự chuyển sang mode LAND!")

            if failsafe_triggered and (alt <= 0.25 or armed_str == "DISARMED") and elapsed > 8.0:
                if armed_str == "DISARMED":
                    landed_and_disarmed = True
                    print(f"\n    [HOÀN TẤT] Drone đã hạ cánh êm ái xuống sàn ({alt:.2f}m) và tự động DISARM ngắt động cơ an toàn!")
                    break

            time.sleep(0.8)

        print("\n" + "=" * 70)
        print("  KẾT QUẢ THỰC NGHIỆM:")
        print(f"  - Phát hiện pin yếu < 10.5V         : {'✅ ĐẠT' if failsafe_triggered else '❌ CHƯA ĐẠT'}")
        print(f"  - Tự động chuyển chế độ LAND         : {'✅ ĐẠT' if failsafe_triggered else '❌ CHƯA ĐẠT'}")
        print(f"  - Hạ cánh thẳng đứng & tự DISARM     : {'✅ ĐẠT' if landed_and_disarmed else '❌ CHƯA ĐẠT'}")
        print("=" * 70)

        # 9. Khôi phục điện áp ban đầu
        mav.mav.param_set_send(
            mav.target_system, mav.target_component,
            b"SIM_BATT_VOLTAGE", 12.6,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )

        return failsafe_triggered and landed_and_disarmed

    finally:
        # Dọn dẹp tiến trình SITL
        print("\n[*] Dọn dẹp và dừng tiến trình SITL...")
        sitl_proc.terminate()
        try:
            sitl_proc.wait(timeout=3.0)
        except Exception:
            sitl_proc.kill()
        print("[+] Hoàn tất.")


if __name__ == "__main__":
    success = run_battery_failsafe_test()
    sys.exit(0 if success else 1)
