#!/usr/bin/env python3
"""
DRONE SWARM ALL-IN-ONE: GAMEPAD LEADER PILOT + 3D APF SWARM ENGINE & EVASION TEST
================================================================================
Kịch bản TẤT-CẢ-TRONG-MỘT (All-in-One):
  1. Người dùng TRỰC TIẾP LÁI DRONE 1 (LEADER) bằng tay cầm vật lý (Xbox, PS, Logitech).
  2. Bầy 4 drone còn lại (Drones 2, 3, 4, 5) TỰ ĐỘNG BÁM ĐUÔI theo hình học đội hình 3D APF.
  3. TỰ ĐỘNG NÉ VA CHẠM APF (Closed-Loop Collision Avoidance):
     - Đọc toạ độ thực tế từ LOCAL_POSITION_NED, tạm dừng an toàn nếu dữ liệu cũ (>0.5s).
     - Khống chế trần bay tối đa 2.0m (bảo vệ ToF MTF-02 2.5m và Geo-Fence phòng kín).
     - Nút X (Button 2): Kích hoạt kịch bản TEST NÉ APF cưỡng bức 6 giây (xung nhiễu cắt mặt) để kiểm chứng live.
  4. Phím chức năng:
     - NÚT A: Arm & Cất cánh đồng bộ toàn bầy (Takeoff All 1.3m)
     - NÚT B: Hạ cánh đồng bộ an toàn (LAND All) - debounce độc lập 0.3s
     - NÚT X: ⚡ Kích hoạt kịch bản TEST NÉ APF
     - NÚT Y: 🛑 Phanh dừng khẩn cấp / Giữ vị trí tại chỗ (BRAKE)
     - NÚT LB / RB: 🔄 Chuyển đổi đội hình (V-Shape ⇄ Diamond ⇄ Line ⇄ Column)
     - CẦN TRÁI: Cao độ Z (Lên/Xuống 0.8m - 2.0m) & Xoay đầu Yaw (Trái/Phải)
     - CẦN PHẢI: Tiến/Lùi X & Lượn ngang Y
================================================================================
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# Chống lỗi font chữ tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.mavlink_handler import MAVLinkHandler, FlightMode
from core.apf_swarm_engine import APFSwarmController, APFConfig, FormationType

# Vùng chết mặc định chống trôi cần
DEADZONE = 0.10

# Tọa độ bãi đỗ xuất phát mặt đất của 5 drone (thu nhỏ cho không gian indoor, span 4.8m)
SWARM_BASE_OFFSETS = {
    0: (0.0, 0.0),    # Drone 1 (Leader - Đỏ):          X_ros = 0.0m, Y_ros = 0.0m
    1: (0.0, 1.2),    # Drone 2 (Follower 1 - Xanh lá): X_ros = 0.0m, Y_ros = 1.2m
    2: (0.0, 2.4),    # Drone 3 (Follower 2 - Xanh lam):X_ros = 0.0m, Y_ros = 2.4m
    3: (0.0, 3.6),    # Drone 4 (Follower 3 - Vàng):    X_ros = 0.0m, Y_ros = 3.6m
    4: (0.0, 4.8),    # Drone 5 (Follower 4 - Tím):     X_ros = 0.0m, Y_ros = 4.8m
}


def normalize_axis(raw_val: float, axis_info: dict = None) -> float:
    """Chuẩn hóa giá trị cần gạt dựa theo điểm tâm nghỉ và biên độ hiệu chuẩn."""
    if not axis_info:
        return 0.0 if abs(raw_val) < DEADZONE else raw_val

    center = axis_info.get("center", 0.0)
    min_v = axis_info.get("min", -1.0)
    max_v = axis_info.get("max", 1.0)
    dz = axis_info.get("deadzone", 0.10)

    diff = raw_val - center
    if abs(diff) < dz:
        return 0.0

    if diff > 0:
        span = max_v - center - dz
        return min(1.0, (diff - dz) / span) if span > 0.01 else 0.0
    else:
        span = center - min_v - dz
        return max(-1.0, (diff + dz) / span) if span > 0.01 else 0.0


def find_joystick():
    import pygame
    pygame.joystick.quit()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        return js
    return None


async def setup_drone(handler: MAVLinkHandler, port: int, drone_id: int, allow_sitl: bool = False) -> bool:
    """Kết nối MAVLink và nạp tham số mượt mà cho 1 drone."""
    conn_str = f"tcp:127.0.0.1:{port}"
    print(f"  [Drone {drone_id}] Đang kết nối tới {conn_str}...")
    connected = await handler.connect(conn_str)
    if not connected:
        print(f"  [Drone {drone_id}] ❌ Lỗi kết nối!")
        return False

    # connect() đã tự chạy telemetry loop ở 10Hz

    try:
        from pymavlink import mavutil
        mav = handler._mavconn.mav
        sys_id = handler._mavconn.target_system or drone_id
        comp_id = handler._mavconn.target_component or 1

        # Chặn ghi các tham số nguy hiểm trừ khi có cờ --sitl tường minh
        if allow_sitl:
            sitl_params = [
                (b'ARMING_CHECK', 0.0),
                (b'AVOID_ENABLE', 0.0),
                (b'DISARM_DELAY', 0.0),
            ]
            for p_name, p_val in sitl_params:
                mav.param_set_send(sys_id, comp_id, p_name, float(p_val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                await asyncio.sleep(0.01)
        else:
            print(f"  [Drone {drone_id}] 🛡️ Chế độ Hardware Safety: Bảo toàn ARMING_CHECK và Failsafe của FC.")

        # Tham số giới hạn tốc độ bay an toàn
        motion_params = [
            (b'WPNAV_SPEED', 120.0),      # Tốc độ lướt mục tiêu: 1.2 m/s
            (b'WPNAV_ACCEL', 100.0),      # Gia tốc: 1.0 m/s^2
            (b'ATC_SLEW_YAW', 4500.0),    # Tốc độ xoay đầu: 45 deg/s
        ]
        for p_name, p_val in motion_params:
            mav.param_set_send(sys_id, comp_id, p_name, float(p_val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"  [Drone {drone_id}] ⚠️ Cảnh báo nạp param: {e}")

    print(f"  [Drone {drone_id}] ✅ Đã kết nối sẵn sàng!")
    return True


async def send_ros_position(handler: MAVLinkHandler, x_ros: float, y_ros: float, z_alt: float, yaw_ros_deg: float = 0.0):
    """
    Chuyển đổi tọa độ ROS ENU (X=East, Y=North, Z=Alt) sang ArduPilot Local NED:
    - X_NED = y_ros (North)
    - Y_NED = x_ros (East)
    - Z_NED = -z_alt (Down)
    - Yaw_NED = (90 - yaw_ros) % 360
    """
    yaw_mav = (90.0 - yaw_ros_deg) % 360.0
    return await handler.set_position_local(x=y_ros, y=x_ros, z_alt=z_alt, yaw=yaw_mav)


def read_actual_poses(active: list, base_offsets: dict) -> Optional[dict]:
    """
    Đọc toạ độ thế giới thực tế từ LOCAL_POSITION_NED của từng drone qua MAVLink (Closed-Loop APF).
    Kiểm tra độ tươi local_timestamp (< 0.5s) và cờ local_valid (Audit R3-01).
    Key toạ độ theo idx cố định (d_id - 1) để tránh lệch khi danh sách active thay đổi (Audit R3-16).
    """
    poses = {}
    now = time.time()
    for d_id, h in active:
        idx = d_id - 1
        t = h.get_telemetry()
        if not t.local_valid or (now - t.local_timestamp) > 0.5:
            return None  # Dữ liệu vị trí cũ hoặc chưa có -> kích hoạt hold_all()
        bx, by = base_offsets[idx]
        x_world = t.local_y + bx
        y_world = t.local_x + by
        z_world = -t.local_z
        yaw_ros = (90.0 - t.yaw) % 360.0
        poses[idx] = (x_world, y_world, z_world, yaw_ros)
    return poses


async def hold_all(active: list, base_offsets: dict, poses: dict):
    """Ghìm chặt toạ độ tại chỗ khi mất telemetry hoặc chờ dữ liệu."""
    tasks = []
    for d_id, h in active:
        idx = d_id - 1
        bx, by = base_offsets[idx]
        if idx in poses:
            px, py, pz, pyaw = poses[idx]
            tasks.append(send_ros_position(h, x_ros=px - bx, y_ros=py - by, z_alt=pz, yaw_ros_deg=pyaw))
    if tasks:
        await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(description="Gamepad Leader 3D APF Swarm Controller")
    parser.add_argument("--sitl", action="store_true", default=False, help="Chế độ mô phỏng SITL (cho phép nạp param bỏ qua ARMING_CHECK)")
    args, _ = parser.parse_known_args()

    print("=" * 72)
    print("   🚀 ALL-IN-ONE: GAMEPAD DRONE 1 LEADER + 3D APF SWARM & EVASION TEST")
    print(f"   Chế độ: {'MÔ PHỎNG SITL' if args.sitl else 'PHẦN CỨNG AN TOÀN (HARDWARE SAFE)'}")
    print("   Lái trực tiếp Drone 1 dẫn đàn | 4 Followers bám đuôi & né va chạm")
    print("=" * 72 + "\n")

    try:
        import pygame
    except ImportError:
        print("❌ Chưa cài đặt thư viện 'pygame'!")
        print("👉 Vui lòng chạy lệnh: python -m pip install pygame")
        return

    pygame.init()
    pygame.joystick.init()

    js = find_joystick()
    if not js:
        print("⏳ CHƯA TÌM THẤY TAY CẦM NÀO ĐƯỢC CẮM!")
        print("👉 Vui lòng CẮM CÁP USB hoặc KẾT NỐI BLUETOOTH tay cầm vào máy tính...")
        while not js:
            await asyncio.sleep(0.5)
            js = find_joystick()

    print(f"🎮 ĐÃ NHẬN DIỆN TAY CẦM: [{js.get_name()}]")
    print(f"   - Số trục Analog: {js.get_numaxes()}")
    print(f"   - Số nút bấm    : {js.get_numbuttons()}")

    # Nạp Profile hiệu chuẩn nếu có
    profile_path = os.path.join(project_root, "config", "gamepad_profile.json")
    profile = None
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                print("📖 Đã nạp cấu hình hiệu chuẩn: [config/gamepad_profile.json]")
        except Exception:
            pass
    axes_cfg = profile.get("axes", {}) if profile else {}

    # 1. Kết nối toàn bộ 5 drone qua SERIAL2 (5773, 5783, 5793, 5803, 5813)
    num_drones = 5
    ports = [5773, 5783, 5793, 5803, 5813]
    handlers = [MAVLinkHandler() for _ in range(num_drones)]

    print(f"\n📡 Đang kết nối tới 5 Drone qua cổng MAVLink {ports}...")
    setup_tasks = [setup_drone(handlers[i], ports[i], i + 1, allow_sitl=args.sitl) for i in range(num_drones)]
    results = await asyncio.gather(*setup_tasks)

    active = [(i + 1, handlers[i]) for i in range(num_drones) if results[i]]
    if len(active) < num_drones:
        print(f"❌ Chỉ kết nối được {len(active)}/5 drones! Hãy đảm bảo SITL hoặc drone đang bật.")
        return

    print(f"\n✅ ĐÃ KẾT NỐI HOÀN TẤT CẢ 5/5 DRONES SẴN SÀNG!")

    # 2. Khởi tạo thuật toán 3D APF Swarm Engine
    apf_config = APFConfig(
        d_influence=0.75,      # (m) Bán kính ảnh hưởng bắt đầu sinh lực đẩy né tránh (chuẩn bầy đàn thu nhỏ)
        d_safe=0.55,           # (m) Khoảng cách an toàn danh định
        d_critical=0.35,       # (m) Khoảng cách khẩn cấp (kích hoạt phản xạ cực đại)
        k_att=1.2,             # Lực hút kéo về đội hình bám sát Leader
        k_rep=2.2,             # Lực đẩy chống va chạm
        v_max=1.2,             # Vận tốc tối đa cho phép của follower
        z_anisotropy=1.5,      # Hệ số an toàn trục Z
    )
    controller = APFSwarmController(apf_config)

    # Danh sách các đội hình hỗ trợ
    formation_list = [
        (FormationType.V_SHAPE, "V-SHAPE (Mũi Tên)"),
        (FormationType.DIAMOND, "DIAMOND (Kim Cương)"),
        (FormationType.LINE,    "LINE (Hàng Ngang)"),
        (FormationType.COLUMN,  "COLUMN (Hàng Dọc)"),
    ]
    formation_idx = 0

    # Biến quản lý trạng thái bay
    is_flying = False
    flight_alt = 1.3  # Cao độ cất cánh mặc định an toàn (1.3m <= 1.5m)

    # Toạ độ thế giới giả định ban đầu của Leader
    leader_x = 3.0
    leader_y = 3.0
    leader_z = flight_alt
    leader_yaw = 0.0

    current_poses = {
        idx: (SWARM_BASE_OFFSETS[idx][0], SWARM_BASE_OFFSETS[idx][1], flight_alt, 0.0)
        for idx in range(num_drones)
    }

    # Tách biến debounce riêng cho từng nút
    last_btn_a_time = 0.0
    last_btn_b_time = 0.0
    last_btn_x_time = 0.0
    last_btn_y_time = 0.0
    last_btn_mode_time = 0.0
    last_health_check = 0.0

    # Quản lý kịch bản Test Né APF
    apf_test_counter = 0
    apf_test_total = 60
    is_braking = False

    TARGET_DT = 0.1  # Chu kỳ 10Hz
    next_t = time.monotonic()

    try:
        while True:
            pygame.event.pump()
            now = time.time()
            t0 = time.monotonic()

            # Đọc các trục Analog (Mode 2 Drone RC chuẩn quốc tế)
            axis_yaw = normalize_axis(js.get_axis(0), axes_cfg.get("axis_0"))
            axis_climb = -normalize_axis(js.get_axis(1), axes_cfg.get("axis_1"))
            axis_roll = normalize_axis(js.get_axis(2) if js.get_numaxes() > 2 else 0.0, axes_cfg.get("axis_2"))
            axis_pitch = -normalize_axis(js.get_axis(3) if js.get_numaxes() > 3 else 0.0, axes_cfg.get("axis_3"))

            # Đọc các nút bấm
            btn_a = js.get_button(0) if js.get_numbuttons() > 0 else 0
            btn_b = js.get_button(1) if js.get_numbuttons() > 1 else 0
            btn_x = js.get_button(2) if js.get_numbuttons() > 2 else 0
            btn_y = js.get_button(3) if js.get_numbuttons() > 3 else 0
            btn_lb = js.get_button(4) if js.get_numbuttons() > 4 else 0
            btn_rb = js.get_button(5) if js.get_numbuttons() > 5 else 0
            btn_back = js.get_button(6) if js.get_numbuttons() > 6 else 0
            btn_start = js.get_button(7) if js.get_numbuttons() > 7 else 0

            # NÚT BACK HOẶC START: THOÁT CHƯƠNG TRÌNH
            if btn_back or btn_start:
                print("\n\n👋 [NÚT BACK/START]: Đã nhận lệnh thoát chương trình từ tay cầm.")
                break

            # NÚT A: CẤT CÁNH ĐỒNG BỘ TOÀN BẦY
            if btn_a and (now - last_btn_a_time > 1.5):
                last_btn_a_time = now
                print(f"\n🛫 [NÚT A]: CẤT CÁNH ĐỒNG BỘ TOÀN BỘ 5 DRONES LÊN {flight_alt}m...")

                for _, h in active:
                    await h.set_mode(FlightMode.GUIDED)

                for attempt in range(6):
                    unarmed = [h for _, h in active if not h.get_telemetry().armed]
                    if not unarmed:
                        break
                    await asyncio.gather(*[h.arm() for h in unarmed])
                    await asyncio.sleep(1.0)

                await asyncio.gather(*[h.takeoff(altitude=flight_alt) for _, h in active])
                print("⏳ Chờ ổn định cao độ trong 3 giây...")
                await asyncio.sleep(3.0)

                leader_x = 1.5
                leader_y = 2.4
                leader_z = flight_alt
                leader_yaw = 0.0
                current_poses[0] = (leader_x, leader_y, leader_z, leader_yaw)
                is_flying = True
                is_braking = False
                print(f"✅ BẦY ĐÀN ĐÃ CẤT CÁNH THÀNH CÔNG! BẮT ĐẦU ĐIỀU KHIỂN LEADER!\n")

            # NÚT B: HẠ CÁNH AN TOÀN (LAND ALL) - Debounce độc lập 0.3s
            if btn_b and (now - last_btn_b_time > 0.3):
                last_btn_b_time = now
                print("\n🛬 [NÚT B]: PHÁT LỆNH HẠ CÁNH AN TOÀN (LAND) ĐỒNG BỘ CHO TOÀN BẦY...")
                is_flying = False
                is_braking = False
                await asyncio.gather(*[h.land() for _, h in active])
                print("✅ Lệnh hạ cánh đã được gửi. Drones đang tiếp đất an toàn.\n")

            # NÚT X: KÍCH HOẠT KỊCH BẢN TEST NÉ APF VẬT LÝ THẬT (R3-17)
            if btn_x and (now - last_btn_x_time > 1.0):
                last_btn_x_time = now
                if not is_flying:
                    print("\n⚠️ Vui lòng bấm NÚT A để cất cánh trước khi test né APF!")
                else:
                    apf_test_counter = apf_test_total
                    print("\n" + "=" * 60)
                    print("🚨 [NÚT X]: KÍCH HOẠT TEST NÉ TRÁNH VA CHẠM APF VẬT LÝ (DRONE 2 TIẾP CẬN DRONE 3)!")
                    print("=" * 60)

            # NÚT Y: TOGGLE BRAKE / GUIDED (R3-15)
            if btn_y and (now - last_btn_y_time > 0.5):
                last_btn_y_time = now
                is_braking = not is_braking
                if is_braking:
                    print(f"\n🛑 [NÚT Y]: PHANH DỪNG KHẨN CẤP! Gửi lệnh BRAKE giữ vị trí cho tất cả drone.")
                    for _, h in active:
                        await h.set_mode(FlightMode.BRAKE)
                else:
                    print(f"\n▶️ [NÚT Y]: NHẢ PHANH! Đồng bộ vị trí Leader với telemetry thực và chuyển lại GUIDED.")
                    actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
                    if actual is not None and 0 in actual:
                        leader_x, leader_y, leader_z, leader_yaw = actual[0]
                    for _, h in active:
                        await h.set_mode(FlightMode.GUIDED)

            # NÚT LB / RB: CHUYỂN ĐỔI ĐỘI HÌNH
            if (btn_lb or btn_rb) and (now - last_btn_mode_time > 0.35):
                last_btn_mode_time = now
                if btn_rb:
                    formation_idx = (formation_idx + 1) % len(formation_list)
                else:
                    formation_idx = (formation_idx - 1) % len(formation_list)
                _, f_name = formation_list[formation_idx]
                print(f"\n🔄 [NÚT LB/RB]: Chuyển sang đội hình: 👉 {f_name}")

            # -------------------------------------------------------------
            # VÒNG LẶP ĐIỀU KHIỂN CHÍNH (10HZ)
            # -------------------------------------------------------------
            if is_flying:
                if is_braking:
                    # Chế độ BRAKE: Bỏ qua tính toán setpoint APF, giữ an toàn tuyệt đối
                    print(
                        f"\r🛑 [BRAKE HOLD ACTIVE] Toàn bầy đang phanh giữ vị trí | Bấm Nút Y lần nữa để nhả phanh...   ",
                        end="",
                    )
                    next_t += TARGET_DT
                    await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                    continue

                # 1. Tính toán vận tốc điều khiển Leader
                v_fwd = axis_pitch * 1.0
                v_strafe = axis_roll * 1.0
                v_climb = axis_climb * 0.5
                v_yaw_rate = axis_yaw * 30.0

                leader_yaw = (leader_yaw + v_yaw_rate * TARGET_DT) % 360.0
                yaw_rad = math.radians(leader_yaw)

                delta_x = (v_fwd * math.cos(yaw_rad) - v_strafe * math.sin(yaw_rad)) * TARGET_DT
                delta_y = (v_fwd * math.sin(yaw_rad) + v_strafe * math.cos(yaw_rad)) * TARGET_DT
                delta_z = v_climb * TARGET_DT

                new_lx = leader_x + delta_x
                new_ly = leader_y + delta_y

                # Soft geofence: Giới hạn bán kính cách Home <= 5.0m (an toàn dưới FENCE_RADIUS 6.0m)
                if math.hypot(new_lx, new_ly) <= 5.0:
                    leader_x = new_lx
                    leader_y = new_ly
                else:
                    print("\n⚠️ [GEOFENCE LIMIT] Leader chạm ranh giới bán kính an toàn 5.0m!", end="")

                # Kẹp cao độ tuyệt đối trong khoảng an toàn 0.8m - 2.0m
                leader_z = max(0.8, min(2.0, leader_z + delta_z))

                # Đóng vòng phản hồi: đọc vị trí thực tế từ LOCAL_POSITION_NED
                actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
                if actual is not None:
                    current_poses = actual
                else:
                    await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                    next_t += TARGET_DT
                    await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                    continue

                # 2. Xử lý kịch bản Test Né APF Vật Lý Thật (R3-17)
                test_alert = ""
                # 3. Tính toán bước di chuyển APF khép vòng với real_dt
                f_type, f_name = formation_list[formation_idx]
                real_dt = max(0.05, min(0.2, time.monotonic() - t0))
                next_cmds, stats = controller.step(
                    current_poses,
                    (leader_x, leader_y, leader_z, leader_yaw),
                    formation_type=f_type,
                    dt=real_dt
                )

                if apf_test_counter > 0:
                    # Ép Drone 2 bay thật về phía Drone 3 trong nửa đầu bài test để kích hoạt né APF thật
                    if apf_test_counter > (apf_test_total // 2):
                        if 1 in next_cmds and 2 in current_poses:
                            p2 = current_poses.get(1, (1.5, 1.2, 1.4, 0.0))
                            # Tiếp cận Drone 3 dọc theo trục Y
                            next_cmds[1] = (p2[0], p2[1] + 0.15, p2[2], p2[3])
                    apf_test_counter -= 1
                    test_alert = "⚡ [TEST NÉ APF VẬT LÝ] "

                # 4. Gửi lệnh toạ độ Local NED tới các drone active (Tránh lỗi trượt index R3-16)
                tasks = []
                for d_id, handler in active:
                    idx = d_id - 1
                    if idx in next_cmds:
                        wx, wy, wz, wyaw = next_cmds[idx]
                        bx, by = SWARM_BASE_OFFSETS[idx]
                        tasks.append(send_ros_position(
                            handler,
                            x_ros=wx - bx,
                            y_ros=wy - by,
                            z_alt=wz,
                            yaw_ros_deg=wyaw
                        ))

                await asyncio.gather(*tasks)

                # 5. Kiểm tra sức khoẻ drone định kỳ (1Hz) - R3-16: Loại bỏ drone lỗi khỏi active
                if now - last_health_check > 1.0:
                    last_health_check = now
                    failed_ids = []
                    for d_id, handler in active:
                        t = handler.get_telemetry()
                        if not t.armed or t.flight_mode not in ("GUIDED", "AUTO", "BRAKE"):
                            print(f"\n⚠️ Drone {d_id} mất an toàn! Mode={t.flight_mode}, Armed={t.armed}")
                            failed_ids.append(d_id)
                    if failed_ids:
                        active = [(d_id, h) for d_id, h in active if d_id not in failed_ids]
                        print(f"⚠️ Cập nhật bầy: {len(active)}/5 drone còn hoạt động.")
                        if len(active) < 2:
                            print("\n🚨 CẢNH BÁO NGUY HIỂM: Số drone hoạt động dưới 2! Tự động hạ cánh khẩn cấp bầy đàn...")
                            is_flying = False
                            await asyncio.gather(*[h.land() for _, h in active])

                # 6. Cập nhật HUD Live trên console
                status_icon = "🟢 NOMINAL"
                if stats['safe_status'] == 'APF_ACTIVE':
                    status_icon = "🟡 APF EVADING"
                elif stats['safe_status'] == 'CRITICAL_AVOID':
                    status_icon = "🔴 CRITICAL AVOID"

                print(
                    f"\r🎮 [LEADER] X:{leader_x:+.1f} Y:{leader_y:+.1f} Z:{leader_z:.1f}m Yaw:{leader_yaw:.0f}° | "
                    f"Đội hình: {f_name.split()[0]} | "
                    f"MinDist: {stats['min_distance_m']}m [{status_icon}] {test_alert}   ",
                    end="",
                )

            else:
                print(
                    f"\r⏳ [CHỜ CẤT CÁNH]: Bấm NÚT A để Arm & Cất cánh bầy 5 Drone lên {flight_alt}m...   ",
                    end="",
                )

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    except KeyboardInterrupt:
        print("\n\n✅ Đã nhận lệnh ngắt Ctrl+C.")
    except Exception as e:
        print(f"\n\n❌ Lỗi trong quá trình thực thi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 [BẢO VỆ AN TOÀN]: Đang kiểm tra và hạ cánh toàn bộ drone trước khi thoát...")
        for d_id, h in active:
            try:
                if h.get_telemetry().armed:
                    print(f"   ↳ Drone {d_id} vẫn ARMED! Phát lệnh LAND khẩn cấp...")
                    await h.land()
            except Exception as e:
                print(f"   ↳ Lỗi gửi land cho Drone {d_id}: {e}")

        # Chờ ngắt động cơ
        disarm_start = time.time()
        while time.time() - disarm_start < 15.0:
            still_armed = [d_id for d_id, h in active if h.get_telemetry().armed]
            if not still_armed:
                break
            await asyncio.sleep(0.5)

        for _, h in active:
            try:
                await h.disconnect()
            except Exception:
                pass
        print("✅ Hoàn tất dọn dẹp kết nối an toàn!")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
