#!/usr/bin/env python3
"""
HOVER ĐỒNG BỘ CẢ 5 DRONE Ở ĐỘ CAO 1.5M (AN TOÀN TRẦN NHÀ & CÓ AUTO-LAND)
1. Kiểm tra tham số và chế độ SITL vs Hardware
2. Chuyển chế độ GUIDED
3. ARM an toàn (Chỉ cho phép Force-ARM khi có cờ --sitl)
4. Phát lệnh TAKEOFF đồng thời (giới hạn <= 1.8m)
5. Giữ trạng thái HOVER, kiểm tra độ cao thực tế (alt >= 1.2m)
6. Tự động LAND toàn bộ đội bay trong khối finally: đảm bảo không treo trên không
"""

import argparse
import sys
import threading
import time
from pymavlink import mavutil

sys.stdout.reconfigure(encoding="utf-8")

DRONES = {
    1: 5773,
    2: 5783,
    3: 5793,
    4: 5803,
    5: 5813,
}


def gcs_heartbeat_worker(connections: dict, stop_event: threading.Event):
    """R3-14: Duy trì GCS Heartbeat 1Hz để chống GCS Failsafe ngắt kết nối."""
    while not stop_event.is_set():
        for mav in list(connections.values()):
            try:
                mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
            except Exception:
                pass
        stop_event.wait(1.0)


def main():
    parser = argparse.ArgumentParser(description="Synchronous 5-Drone Hover Test with Safety Landing")
    parser.add_argument("--sitl", action="store_true", help="Chế độ SITL (cho phép reset pin giả lập và cờ force-arm)")
    parser.add_argument("--alt", type=float, default=1.5, help="Độ cao mục tiêu (m, mặc định 1.5m, kẹp <= 1.8m)")
    parser.add_argument("--duration", type=float, default=6.0, help="Thời gian duy trì hover (giây, mặc định 6s)")
    args = parser.parse_args()

    # Kẹp độ cao an toàn trần nhà
    target_alt = min(max(args.alt, 0.8), 1.8)

    print("=" * 65)
    print(f"  🚀 BẮT ĐẦU QUY TRÌNH TEST HOVER ĐỒNG BỘ CẢ 5 DRONE ({target_alt:.2f}M)")
    print(f"  Chế độ: {'SITL Giả Lập' if args.sitl else 'PHẦN CỨNG THẬT (STRICT SAFETY)'}")
    print("=" * 65)

    conns = {}
    # Kết nối toàn bộ drone
    for drone_id, port in DRONES.items():
        try:
            mav = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", timeout=3)
            mav.wait_heartbeat(timeout=3)
            conns[drone_id] = mav
            print(f"  🔗 Drone {drone_id} (port {port}): Kết nối MAVLink thành công (Target Sys:{mav.target_system}, Comp:{mav.target_component}).")
        except Exception as e:
            print(f"  ❌ Drone {drone_id} (port {port}): Lỗi kết nối: {e}")

    if len(conns) < len(DRONES):
        print("⚠️ Cảnh báo: Một số drone không kết nối được!")

    # R3-14: Bật luồng gửi GCS Heartbeat 1Hz liên tục
    stop_hb = threading.Event()
    hb_thread = threading.Thread(target=gcs_heartbeat_worker, args=(conns, stop_hb), daemon=True)
    hb_thread.start()

    try:
        # BƯỚC 1: CẤU HÌNH & CHUYỂN GUIDED
        print("\n[BƯỚC 1/4] ⚙️ Cấu hình trạng thái & Chuyển chế độ GUIDED...")
        for drone_id, mav in conns.items():
            try:
                if args.sitl:
                    mav.param_set_send("SIM_BATT_CAP_AH", 1.41)
                    mav.param_set_send("SIM_BATT_VOLTAGE", 12.61)
                    time.sleep(0.04)
                    mav.param_set_send("SIM_BATT_CAP_AH", 1.40)
                    mav.param_set_send("SIM_BATT_VOLTAGE", 12.60)
                    mav.param_set_send("ARMING_CHECK", 0)
                    print(f"  ⚡ Drone {drone_id}: [SITL] Đã nạp đầy 12.6V & Tắt ARMING_CHECK")
                else:
                    print(f"  🛡️ Drone {drone_id}: [HARDWARE] Giữ nguyên tham số EEPROM và ARMING_CHECK an toàn")

                mav.set_mode("GUIDED")
                print(f"  🕹️ Drone {drone_id}: Đã gửi lệnh GUIDED")
            except Exception as e:
                print(f"  ❌ Drone {drone_id}: Lỗi bước 1: {e}")

        time.sleep(0.8)

        # BƯỚC 2: ARM ĐỒNG BỘ CẢ 5 DRONE (R3-14: Sử dụng mav.target_system và mav.target_component)
        print("\n[BƯỚC 2/4] ⚔️ Khởi động quay động cơ (ARM) cả 5 drone...")
        arm_param2 = 21196 if args.sitl else 0
        for drone_id, mav in conns.items():
            try:
                mav.mav.command_long_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    1, arm_param2,
                    0, 0, 0, 0, 0
                )
                print(f"  🔥 Drone {drone_id}: Đã phát lệnh ARM (param2={arm_param2})")
            except Exception as e:
                print(f"  ❌ Drone {drone_id}: Lỗi ARM: {e}")

        print("⏳ Chờ 1.5 giây để động cơ cả 5 drone quay ổn định...")
        time.sleep(1.5)

        # BƯỚC 3: PHÁT LỆNH CẤT CÁNH (R3-14: Sử dụng mav.target_system và mav.target_component)
        print(f"\n[BƯỚC 3/4] 🛫 Phát lệnh CẤT CÁNH đồng loạt lên {target_alt}m...")
        for drone_id, mav in conns.items():
            try:
                mav.mav.command_long_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0, 0, 0, 0, 0, 0,
                    float(target_alt)
                )
                print(f"  🛫 Drone {drone_id}: Đã gửi lệnh NAV_TAKEOFF {target_alt}m")
            except Exception as e:
                print(f"  ❌ Drone {drone_id}: Lỗi TAKEOFF: {e}")

        # BƯỚC 4: DUY TRÌ HOVER & KIỂM TRA ĐỘ CAO THỰC TẾ
        print(f"\n[BƯỚC 4/4] 📊 Đang duy trì Hover trong {args.duration}s & đo đạc cao độ...")
        time.sleep(3.0)

        # Yêu cầu data stream
        for drone_id, mav in conns.items():
            try:
                mav.mav.request_data_stream_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
                )
            except Exception:
                pass

        time.sleep(max(1.0, args.duration - 3.0))

        print("=" * 65)
        print(f"  KẾT QUẢ ĐO ĐẠC THỰC TẾ TRÊN GCS & SITL (HOVERING {target_alt}M):")
        print("=" * 65)

        heights = {}
        min_reach_alt = target_alt - 0.3  # Ngưỡng tối thiểu công nhận đã cất cánh đạt
        for drone_id, mav in conns.items():
            alt = 0.0
            thr = 0
            volt = 0.0
            try:
                pos = mav.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
                vfr = mav.recv_match(type='VFR_HUD', blocking=True, timeout=1.0)
                bat = mav.recv_match(type='SYS_STATUS', blocking=True, timeout=1.0)
                alt = -pos.z if pos else (vfr.alt if vfr else 0.0)
                volt = bat.voltage_battery / 1000.0 if bat else 0.0
                thr = vfr.throttle if vfr else 0
                heights[drone_id] = alt
                print(f"  🚁 DRONE {drone_id}: Cao độ = {alt:4.2f}m | Ga = {thr:2d}% | Pin = {volt:4.1f}V | Trạng thái = GUIDED HOVER")
            except Exception as e:
                print(f"  ❌ Drone {drone_id}: {e}")

        # Kiểm tra ngưỡng thành công
        all_reached = len(heights) == len(DRONES) and all(h >= min_reach_alt for h in heights.values())
        print("=" * 65)
        if all_reached:
            print(f"  🎉 TOÀN BỘ 5 DRONE ĐÃ LƠ LỬNG ỔN ĐỊNH Ở {target_alt}M (Đo đạc: >= {min_reach_alt:.2f}m)!")
        else:
            failed = [d for d, h in heights.items() if h < min_reach_alt]
            print(f"  ⚠️ Cảnh báo: Một số drone chưa đạt cao độ mục tiêu: {failed}")
        print("=" * 65)

    except KeyboardInterrupt:
        print("\n⚠️ Người dùng bấm Ctrl+C: Hủy quá trình hover!")
    except Exception as e:
        print(f"\n❌ Lỗi bất thường trong quá trình hover: {e}")
    finally:
        # Dừng luồng GCS Heartbeat
        stop_hb.set()
        hb_thread.join(timeout=1.5)

        # TỰ ĐỘNG HẠ CÁNH AN TOÀN TOÀN BỘ ĐỘI BAY
        print("\n" + "=" * 65)
        print("  🛬 [AUTO-LAND FINALLY] ĐỒNG BỘ HẠ CÁNH AN TOÀN CHO TOÀN BỘ DRONE...")
        print("=" * 65)
        for drone_id, mav in conns.items():
            try:
                mav.set_mode("LAND")
                print(f"  🛬 Drone {drone_id}: Đã gửi lệnh LAND")
            except Exception as e:
                print(f"  ❌ Drone {drone_id}: Lỗi gửi lệnh LAND: {e}")

        # Chờ disarm với timeout 15 giây (R3-13: Xác nhận khẳng định bằng tập hợp disarmed_ids)
        print("⏳ Đang theo dõi quá trình hạ cánh và ngắt động cơ (timeout 15s)...")
        start_wait = time.time()
        disarmed_ids = set()
        while time.time() - start_wait < 15.0:
            for drone_id, mav in conns.items():
                if drone_id in disarmed_ids:
                    continue
                try:
                    while True:
                        hb = mav.recv_match(type='HEARTBEAT', blocking=False)
                        if not hb:
                            break
                        armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        if not armed:
                            disarmed_ids.add(drone_id)
                            print(f"  🛑 Drone {drone_id}: Đã xác nhận DISARMED thành công.")
                            break
                except Exception:
                    pass
            if len(disarmed_ids) == len(conns) and len(conns) > 0:
                print("  ✅ Toàn bộ drone đã xác nhận hạ cánh an toàn và ngắt động cơ (DISARMED)!")
                break
            time.sleep(1.0)

        still_active = [d for d in conns if d not in disarmed_ids]
        if still_active:
            print(f"  ⚠️ Cảnh báo: Các drone chưa xác nhận disarm: {still_active}")

        # Đóng tất cả kết nối MAVLink
        for drone_id, mav in conns.items():
            try:
                mav.close()
            except Exception:
                pass
        print("🔒 Đã giải phóng toàn bộ kết nối MAVLink. Hoàn tất!")

if __name__ == "__main__":
    main()
