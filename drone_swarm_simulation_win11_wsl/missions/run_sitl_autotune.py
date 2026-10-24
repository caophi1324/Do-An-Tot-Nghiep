#!/usr/bin/env python3
"""
DRONEEXE - ArduPilot SITL Real-World AutoTune Runner
=============================================================================
Khởi chạy quy trình AutoTune chính thức của ArduPilot trong môi trường mô phỏng SITL:
1. Cất cánh Drone lên 1.8m.
2. Chuyển sang ALT_HOLD kèm RC Override (1500 mid-stick).
3. Kích hoạt Flight Mode AUTOTUNE (Mode 15).
4. Lắng nghe và hiển thị toàn bộ tiến trình AutoTune (Roll Rate D/P, Pitch Rate D/P, Angle P).
5. Khi AutoTune báo hoàn thành (AutoTune: Success), tự động hạ cánh LAND.
6. Đọc trực tiếp bộ tham số PID tối ưu vừa được ArduPilot tự tính toán!
=============================================================================
"""

import time
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pymavlink import mavutil

def run_autotune():
    print("=====================================================================")
    print("🚀 BẮT ĐẦU CHẠY AUTOTUNE THỰC TẾ TRONG ARDUPILOT SITL (DRONE 1)")
    print("=====================================================================")
    
    print("\n📡 Đang kết nối MAVLink tới Drone 1 (tcp:127.0.0.1:5773)...")
    mav = mavutil.mavlink_connection('tcp:127.0.0.1:5773')
    mav.wait_heartbeat()
    print("✅ Đã kết nối MAVLink thành công!")

    # 1. Khởi tạo toạ độ Home / Origin
    mav.mav.set_gps_global_origin_send(
        mav.target_system,
        int(-35.363261 * 1e7),
        int(149.165230 * 1e7),
        int(584.0 * 1000)
    )
    time.sleep(0.5)

    # 2. Chuyển sang GUIDED để cất cánh
    guided_mode = mav.mode_mapping()['GUIDED']
    mav.set_mode(guided_mode)
    time.sleep(0.5)

    # 3. ARM động cơ
    print("\n[1/5] ARM động cơ...")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(1.0)

    # 4. Cất cánh lên 1.8m
    target_alt = 1.8
    print(f"[2/5] Cất cánh lên {target_alt}m...")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_alt
    )

    t0 = time.time()
    while time.time() - t0 < 10:
        msg = mav.recv_match(type=['GLOBAL_POSITION_INT', 'STATUSTEXT'], blocking=True, timeout=1.0)
        if msg:
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                alt = msg.relative_alt / 1000.0
                if alt >= 1.65:
                    print(f"  -> Đạt cao độ an toàn: {alt:.2f}m")
                    break
            elif msg.get_type() == 'STATUSTEXT':
                print(f"  [STATUSTEXT] {msg.text}")

    print("\n[3/5] Duy trì lơ lửng ổn định, gửi RC Override 1500...")
    for _ in range(3):
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            1500, 1500, 1500, 1500, 0, 0, 0, 0
        )
        time.sleep(0.3)

    # Chuyển sang ALT_HOLD (bắt buộc trước khi vào AUTOTUNE)
    print("  -> Chuyển sang ALT_HOLD (Mode 2)...")
    mav.set_mode(2)
    time.sleep(1.5)

    # 5. Kích hoạt AUTOTUNE (Mode 15)
    print("\n[4/5] Kích hoạt chế độ AUTOTUNE (Mode 15) của ArduPilot...")
    mav.set_mode(15)

    autotune_done = False
    start_tune = time.time()
    max_duration = 180  # Tối đa 3 phút

    last_log_time = 0
    while time.time() - start_tune < max_duration:
        # Giữ RC Override để ArduPilot không báo mất sóng RC
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            1500, 1500, 1500, 1500, 0, 0, 0, 0
        )
        
        msg = mav.recv_match(blocking=True, timeout=0.5)
        if msg:
            mtype = msg.get_type()
            if mtype == 'STATUSTEXT':
                txt = msg.text
                print(f"  🔔 [ArduPilot AutoTune] {txt}")
                if "success" in txt.lower() or "complete" in txt.lower():
                    autotune_done = True
                    break
            elif mtype == 'GLOBAL_POSITION_INT':
                now = time.time()
                if now - last_log_time >= 5.0:
                    alt = msg.relative_alt / 1000.0
                    vx = msg.vx / 100.0
                    vy = msg.vy / 100.0
                    print(f"  [Tele] Cao độ: {alt:.2f}m | V_xy: ({vx:.2f}, {vy:.2f}) m/s | Thời gian: {int(now - start_tune)}s")
                    last_log_time = now

    if autotune_done:
        print("\n🎉 AUTOTUNE HOÀN TẤT THÀNH CÔNG!")
    else:
        print(f"\n⚠️ Kết thúc phiên AutoTune sau {int(time.time() - start_tune)} giây.")

    # 6. Hạ cánh LAND
    print("\n[5/5] Ra lệnh hạ cánh an toàn (LAND)...")
    land_mode = mav.mode_mapping()['LAND']
    mav.set_mode(land_mode)
    time.sleep(5.0)

    # 7. Đọc thông số PID vừa được tune
    print("\n=====================================================================")
    print("📊 BỘ THAM SỐ PID DO ARDUPILOT AUTOTUNE TỰ ĐỘNG XÁC ĐỊNH:")
    print("=====================================================================")
    params_to_read = [
        'ATC_RAT_RLL_P', 'ATC_RAT_RLL_I', 'ATC_RAT_RLL_D', 'ATC_ANG_RLL_P',
        'ATC_RAT_PIT_P', 'ATC_RAT_PIT_I', 'ATC_RAT_PIT_D', 'ATC_ANG_PIT_P',
        'ATC_ACCEL_R_MAX', 'ATC_ACCEL_P_MAX'
    ]
    
    for p in params_to_read:
        mav.mav.param_request_read_send(mav.target_system, mav.target_component, p.encode('utf-8'), -1)
        t_param = time.time()
        while time.time() - t_param < 1.0:
            pmsg = mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
            if pmsg and pmsg.param_id.strip('\x00') == p:
                print(f"  - {p:18s} : {pmsg.param_value:.6f}")
                break

    print("=====================================================================\n")

if __name__ == "__main__":
    run_autotune()
