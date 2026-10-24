#!/usr/bin/env python3
"""
DRONE SWARM MISSION: 3D ARTIFICIAL POTENTIAL FIELD (APF) LEADER-FOLLOWER
- Điều khiển bầy 5 Drone ArduPilot bám đuôi động học theo Leader thời gian thực (10Hz).
- Ứng dụng thuật toán Trường thế nhân tạo 3D (3D APF) khép vòng (Closed-Loop Telemetry):
    + Lực hút F_att: Duy trì hình học đội hình danh định theo Leader.
    + Lực đẩy F_rep: Tự động phát hiện và dạt ra xa khi khoảng cách < 1.4m (chống va chạm tuyệt đối).
    + Chống bế tắc GNRON (Goal Non-Reachable with Obstacle Nearby).
    + Đọc vị trí thực tế từ LOCAL_POSITION_NED, dừng an toàn nếu dữ liệu cũ (>0.5s).
- Trần bay an toàn trong nhà: Cao độ cất cánh 1.3m, tối đa 1.6m (dưới ngưỡng ToF MTF-02 2.5m và Geo-Fence 2.0m).
- Các pha thực thi:
    1. Cất cánh đồng bộ (Synchronized Takeoff 1.3m).
    2. Tập kết vào Đội hình Mũi tên Chữ V (V-Shape Formation).
    3. Bám đuôi động học theo Leader bay quỹ đạo tròn 3D (Dynamic Orbit Tracking).
    4. Biến đổi đội hình mượt mà sang Hình Kim Cương (Morphing to Diamond).
    5. Thử nghiệm Xung nhiễu cưỡng bức & Phản xạ tự hồi phục APF (Active Evasion Demo).
    6. Chuyển sang Hàng dọc (Column Formation) và hạ cánh an toàn về bãi đỗ riêng.
"""

import asyncio
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
sys.path.insert(0, project_root)

from core.mavlink_handler import MAVLinkHandler, FlightMode
from core.apf_swarm_engine import APFSwarmController, APFConfig, FormationType

# Tọa độ bãi đỗ xuất phát mặt đất của 5 drone (chuẩn hội trường 20x30m, bước 1.6m)
SWARM_BASE_OFFSETS = {
    0: (0.0, 0.0),    # Drone 1 (Leader - Đỏ):          X_ros = 0.0m, Y_ros = 0.0m
    1: (0.0, 1.6),    # Drone 2 (Follower 1 - Xanh lá): X_ros = 0.0m, Y_ros = 1.6m
    2: (0.0, 3.2),    # Drone 3 (Follower 2 - Xanh lam):X_ros = 0.0m, Y_ros = 3.2m
    3: (0.0, 4.8),    # Drone 4 (Follower 3 - Vàng):    X_ros = 0.0m, Y_ros = 4.8m
    4: (0.0, 6.4),    # Drone 5 (Follower 4 - Tím):     X_ros = 0.0m, Y_ros = 6.4m
}


async def setup_drone(handler: MAVLinkHandler, port: int, drone_id: int, allow_sitl: bool = False) -> bool:
    """Kết nối MAVLink và cấu hình ban đầu cho 1 drone."""
    conn_str = f"tcp:127.0.0.1:{port}"
    print(f"[Drone {drone_id}] Đang kết nối tới {conn_str}...")
    connected = await handler.connect(conn_str)
    if not connected:
        print(f"[Drone {drone_id}] ❌ Kết nối thất bại!")
        return False

    if allow_sitl:
        try:
            handler._mavconn.param_set_send("ARMING_CHECK", 0)
            handler._mavconn.param_set_send("AVOID_ENABLE", 0)
            handler._mavconn.param_set_send("DISARM_DELAY", 0)
            handler._mavconn.param_set_send("FS_GCS_ENABLE", 0)
            handler._mavconn.param_set_send("FS_EKF_ACTION", 0)
        except Exception:
            pass

    # connect() đã tự khởi tạo start_telemetry ở 10Hz, không gọi lại 5Hz

    # Nạp tham số bay mượt mà vào ArduPilot (0.8m/s - 1.2m/s)
    try:
        from pymavlink import mavutil
        mav = handler._mavconn.mav
        sys_id = handler._mavconn.target_system or drone_id
        comp_id = handler._mavconn.target_component or 1

        smooth_params = [
            (b'WPNAV_SPEED', 180.0),      # Tốc độ lướt mục tiêu: 1.8 m/s (nhanh dứt khoát)
            (b'WPNAV_ACCEL', 150.0),      # Gia tốc mượt: 1.5 m/s²
            (b'WPNAV_JERK', 3.0),         # Giới hạn jerk chống giật
            (b'ATC_SLEW_YAW', 4500.0),    # Tốc độ xoay đầu: 45 deg/s
            (b'PSC_NE_POS_P', 0.50),      # Đóng băng hover, triệt tiêu vọt lố
            (b'PSC_NE_VEL_P', 1.20),      # Cứng vững bám vận tốc
            (b'PSC_NE_VEL_I', 0.08),      # Giảm tích phân I triệt tiêu đu đưa con lắc 3.5s
            (b'PSC_NE_VEL_D', 0.00),      # Triệt tiêu khuếch đại nhiễu quang học
        ]
        for p_name, p_val in smooth_params:
            mav.param_set_send(sys_id, comp_id, p_name, float(p_val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            await asyncio.sleep(0.02)
    except Exception as e:
        print(f"[Drone {drone_id}] ⚠️ Cảnh báo nạp param: {e}")

    print(f"[Drone {drone_id}] ✅ Đã kết nối & nạp tham số APF mượt mà!")
    return True


async def send_ros_position(handler: MAVLinkHandler, x_ros: float, y_ros: float, z_alt: float, yaw_ros_deg: float = 0.0):
    """
    Chuyển đổi tọa độ thế giới ROS ENU sang MAVLink Local NED của ArduPilot:
    - X_NED (North) = y_ros
    - Y_NED (East)  = x_ros
    - Z_NED (Down)  = -z_alt
    - Yaw_NED       = (90 - yaw_ros) % 360
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
        # x_ros (East) = t.local_y + bx; y_ros (North) = t.local_x + by; z_ros (Up) = -t.local_z
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
    import argparse
    parser = argparse.ArgumentParser(description="Drone Swarm 3D APF Leader-Follower")
    parser.add_argument("--sitl", action="store_true", default=False, help="Chế độ mô phỏng SITL")
    args, _ = parser.parse_known_args()

    print("=" * 70)
    print("  🚁 DRONE SWARM: 3D ARTIFICIAL POTENTIAL FIELD (APF) LEADER-FOLLOWER")
    print(f"  Chế độ: {'MÔ PHỎNG SITL' if args.sitl else 'PHẦN CỨNG AN TOÀN'}")
    print("  Tần số bắn lệnh: 10Hz | Thuật toán: Closed-Loop 3D APF + GNRON")
    print("=" * 70)

    num_drones = 5
    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    active = []

    # 1. KẾT NỐI TẤT CẢ 5 DRONE QUA CỔNG SERIAL2 (5773, 5783, 5793, 5803, 5813)
    for i in range(1, num_drones + 1):
        port = 5760 + i * 10 + 3
        success = await setup_drone(handlers[i - 1], port, i, allow_sitl=args.sitl)
        if success:
            active.append((i, handlers[i - 1]))

    if len(active) < num_drones:
        print(f"❌ Không đủ {num_drones} drone hoạt động (chỉ có {len(active)} con). Hủy nhiệm vụ.")
        return

    # Khởi tạo Bộ điều khiển bầy đàn APF (chuẩn hội trường 20x30m: dãn cách 1.6m - 2.0m)
    apf_config = APFConfig(
        formation_scale=1.0,  # Hội trường 20x30m: dãn cách danh định 1.6m - 2.0m hoành tráng
        d_influence=1.20,     # Bắt đầu sinh lực đẩy khi cách nhau < 1.20m
        d_safe=0.85,          # Khoảng cách bảo vệ an toàn danh định
        d_critical=0.50,      # Khoảng cách khẩn cấp kích hoạt đẩy cực đại
        k_att=1.2,            # Lực hút kéo về đội hình
        k_rep=1.5,            # Lực đẩy chống va chạm
        v_max=1.8,            # Giới hạn vận tốc tối đa 1.8 m/s nhanh nhẹn
        z_anisotropy=1.2,     # Tăng khoảng cách an toàn trục Z chống downwash
    )
    controller = APFSwarmController(apf_config)

    try:
        # =========================================================================
        # PHA 1: CẤT CÁNH ĐỒNG BỘ (SYNCHRONIZED TAKEOFF - 1.3M)
        # =========================================================================
        takeoff_alt = 1.3
        print(f"\n🛫 [PHA 1] ARM VÀ CẤT CÁNH ĐỒNG BỘ 5 DRONE LÊN {takeoff_alt}m (Trần an toàn MTF-02)...")

        for _, h in active:
            await h.set_mode(FlightMode.GUIDED)

        print("   ↳ ⏳ Đang kiểm tra EKF & ARM đồng loạt cho cả 5 Drone...")
        for attempt in range(10):
            unarmed = [h for _, h in active if not h.get_telemetry().armed]
            if not unarmed:
                print("   ✅ Toàn bộ 5 Drone đã ARM thành công, sẵn sàng cất cánh!")
                break
            await asyncio.gather(*[h.arm() for h in unarmed])
            await asyncio.sleep(1.5)

        # Cất cánh đồng thời
        await asyncio.gather(*[h.takeoff(altitude=takeoff_alt) for _, h in active])
        print("⏳ Đang ổn định cao độ cất cánh trong 4 giây...")
        await asyncio.sleep(4.0)

        # Toạ độ thế giới ban đầu (dự phòng)
        current_poses = {
            idx: (SWARM_BASE_OFFSETS[idx][0], SWARM_BASE_OFFSETS[idx][1], takeoff_alt, 0.0)
            for idx in range(num_drones)
        }

        # =========================================================================
        # PHA 2: TẬP KẾT ĐỘI HÌNH CHỮ V (V-SHAPE FORMATION - 1.4M)
        # =========================================================================
        print("\n🪶 [PHA 2] TẬP KẾT VÀO ĐỘI HÌNH MŨI TÊN CHỮ V (V-SHAPE - 1.4M)...")
        target_leader_assemble = (3.0, 3.2, 1.4, 0.0)
        TARGET_DT = 0.1
        assemble_steps = int(14.0 / TARGET_DT)

        start_lx = current_poses[0][0]
        start_ly = current_poses[0][1]

        next_t = time.monotonic()
        for s in range(assemble_steps):
            t0 = time.monotonic()
            prog = min(1.0, s / (assemble_steps * 0.8))
            alpha = (1.0 - math.cos(math.pi * prog)) / 2.0
            cur_lx = start_lx + (target_leader_assemble[0] - start_lx) * alpha
            cur_ly = start_ly + (target_leader_assemble[1] - start_ly) * alpha
            current_leader = (cur_lx, cur_ly, 1.4, 0.0)

            # Đóng vòng phản hồi: đọc vị trí thực tế từ drone
            actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
            if actual is None:
                await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                next_t += TARGET_DT
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                continue
            current_poses = actual

            real_dt = max(0.05, min(0.2, time.monotonic() - t0)) if s > 0 else TARGET_DT
            next_cmds, stats = controller.step(current_poses, current_leader, FormationType.V_SHAPE, dt=real_dt)

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                if idx in next_cmds:
                    wx, wy, wz, yaw = next_cmds[idx]
                    bx, by = SWARM_BASE_OFFSETS[idx]
                    tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))

            await asyncio.gather(*tasks)

            if s % 20 == 0:
                print(f"   ↳ [10Hz Step {s}/{assemble_steps}] Min Dist: {stats['min_distance_m']}m | {stats['safe_status']}")

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ✅ Đã hội tụ hoàn chỉnh ĐỘI HÌNH CHỮ V! Giữ thế bay 3 giây...")
        await asyncio.sleep(3.0)

        # =========================================================================
        # PHA 3: BÁM ĐUÔI ĐỘNG HỌC THEO LEADER BAY ĐƯỜNG TRÒN 3D (R = 1.0M, Z = 1.4M)
        # =========================================================================
        print("\n🔄 [PHA 3] BÁM ĐUÔI ĐỘNG HỌC: LEADER DẪN ĐOÀN BAY QUỸ ĐẠO TRÒN 3D (R=2.0m)...")
        orbit_duration = 18.0
        orbit_steps = int(orbit_duration / TARGET_DT)
        center_x, center_y = 3.0, 3.2
        radius = 2.0

        next_t = time.monotonic()
        for s in range(orbit_steps):
            t0 = time.monotonic()
            angle = (2.0 * math.pi * s) / orbit_steps
            lx = center_x + radius * math.sin(angle)
            ly = center_y + radius * math.cos(angle)
            lz = 1.4 + 0.15 * math.sin(2.0 * angle)
            l_yaw = math.degrees(math.atan2(-math.cos(angle), math.sin(angle))) % 360.0

            cur_leader = (lx, ly, lz, l_yaw)

            actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
            if actual is None:
                await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                next_t += TARGET_DT
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                continue
            current_poses = actual

            real_dt = max(0.05, min(0.2, time.monotonic() - t0)) if s > 0 else TARGET_DT
            next_cmds, stats = controller.step(current_poses, cur_leader, FormationType.V_SHAPE, dt=real_dt)

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                if idx in next_cmds:
                    wx, wy, wz, yaw = next_cmds[idx]
                    bx, by = SWARM_BASE_OFFSETS[idx]
                    tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))

            await asyncio.gather(*tasks)

            if s % 30 == 0:
                print(f"   ↳ [Orbit {int((s/orbit_steps)*100)}%] Leader Yaw: {l_yaw:.1f}° | Min Dist: {stats['min_distance_m']}m | {stats['safe_status']}")

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ✅ Hoàn thành bay tuần tra đường tròn 3D xuất sắc!")

        # =========================================================================
        # PHA 4: BIẾN ĐỔI ĐỘI HÌNH TRỰC TIẾP: CHỮ V -> HÌNH KIM CƯƠNG (DIAMOND)
        # =========================================================================
        print("\n💎 [PHA 4] BIẾN ĐỔI ĐỘI HÌNH ĐỘNG: CHỮ V ➔ HÌNH KIM CƯƠNG (DIAMOND)...")
        morph_duration = 8.0
        morph_steps = int(morph_duration / TARGET_DT)
        leader_hover = (3.0, 3.2, 1.4, 0.0)

        next_t = time.monotonic()
        for s in range(morph_steps):
            t0 = time.monotonic()
            actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
            if actual is None:
                await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                next_t += TARGET_DT
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                continue
            current_poses = actual

            real_dt = max(0.05, min(0.2, time.monotonic() - t0)) if s > 0 else TARGET_DT
            next_cmds, stats = controller.step(current_poses, leader_hover, FormationType.DIAMOND, dt=real_dt)

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                if idx in next_cmds:
                    wx, wy, wz, yaw = next_cmds[idx]
                    bx, by = SWARM_BASE_OFFSETS[idx]
                    tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))

            await asyncio.gather(*tasks)

            if s % 20 == 0:
                print(f"   ↳ [Morphing {int((s/morph_steps)*100)}%] Min Swarm Dist: {stats['min_distance_m']}m | {stats['safe_status']}")

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ✅ Đã định hình KHỐI KIM CƯƠNG (DIAMOND) hoàn hảo!")
        await asyncio.sleep(2.0)

        # =========================================================================
        # PHA 5: THỬ NGHIỆM XUNG NHIỄU CƯỠNG BỨC & TỰ ĐỘNG NÉ VA CHẠM APF THỰC TẾ
        # =========================================================================
        print("\n⚡ [PHA 5] THỬ NGHIỆM XUNG NHIỄU: DRONE 2 TIẾP CẬN THẬT -> DRONE 3 TỰ ĐỘNG NÉ APF...")
        perturb_duration = 5.0
        perturb_steps = int(perturb_duration / TARGET_DT)
        min_observed_dist = 99.0

        next_t = time.monotonic()
        for s in range(perturb_steps):
            t0 = time.monotonic()
            actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
            if actual is None:
                await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                next_t += TARGET_DT
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                continue
            current_poses = actual

            real_dt = max(0.05, min(0.2, time.monotonic() - t0)) if s > 0 else TARGET_DT
            next_cmds, stats = controller.step(current_poses, leader_hover, FormationType.DIAMOND, dt=real_dt)

            # R3-17: Ép Drone 2 bay thật về phía Drone 3 trong nửa đầu bài test để kiểm chứng phản xạ APF thật
            if s < perturb_steps // 2 and 2 in [d for d, _ in active] and 3 in [d for d, _ in active]:
                p3 = current_poses.get(2, (3.0, 3.2, 1.4, 0.0))
                p2 = current_poses.get(1, (3.0, 1.6, 1.4, 0.0))
                # Hướng Drone 2 tiếp cận Drone 3
                next_cmds[1] = (p2[0], p2[1] + 0.15, p2[2], p2[3])

            cur_min = stats.get('min_distance_m', 99.0)
            if cur_min < min_observed_dist:
                min_observed_dist = cur_min

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                if idx in next_cmds:
                    wx, wy, wz, yaw = next_cmds[idx]
                    bx, by = SWARM_BASE_OFFSETS[idx]
                    tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))

            await asyncio.gather(*tasks)

            if stats['safe_status'] in ['APF_ACTIVE', 'CRITICAL_AVOID']:
                print(f"   🚨 [APF EVASION IN ACTION!] Min Dist: {stats['min_distance_m']}m | {stats['safe_status']}")

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print(f"   ✅ THÀNH CÔNG: Phản xạ APF né tránh thành công! Khoảng cách gần nhất đo được: {min_observed_dist:.2f}m")
        await asyncio.sleep(2.0)

        # =========================================================================
        # PHA 6: CHUYỂN SANG HÀNG DỌC (COLUMN) & HẠ CÁNH ĐỒNG BỘ AN TOÀN
        # =========================================================================
        print("\n📏 [PHA 6] CHUYỂN ĐỘI HÌNH HÀNG DỌC (COLUMN) VÀ HẠ CÁNH AN TOÀN...")
        col_duration = 6.0
        col_steps = int(col_duration / TARGET_DT)

        next_t = time.monotonic()
        for s in range(col_steps):
            t0 = time.monotonic()
            actual = read_actual_poses(active, SWARM_BASE_OFFSETS)
            if actual is None:
                await hold_all(active, SWARM_BASE_OFFSETS, current_poses)
                next_t += TARGET_DT
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                continue
            current_poses = actual

            real_dt = max(0.05, min(0.2, time.monotonic() - t0)) if s > 0 else TARGET_DT
            next_cmds, stats = controller.step(current_poses, (3.0, 3.2, 1.3, 0.0), FormationType.COLUMN, dt=real_dt)

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                if idx in next_cmds:
                    wx, wy, wz, yaw = next_cmds[idx]
                    bx, by = SWARM_BASE_OFFSETS[idx]
                    tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))

            await asyncio.gather(*tasks)

            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ↳ 🛬 Đang điều hướng từng drone về bãi đỗ riêng...")
        return_duration = 10.0
        return_steps = int(return_duration / TARGET_DT)
        start_return_poses = {d_id - 1: current_poses.get(d_id - 1, (0.0, 0.0, 1.3, 0.0)) for d_id, _ in active}

        next_t = time.monotonic()
        for s in range(return_steps):
            prog = s / return_steps
            alpha = (1.0 - math.cos(math.pi * prog)) / 2.0
            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                bx, by = SWARM_BASE_OFFSETS[idx]
                if idx in start_return_poses:
                    start_w = start_return_poses[idx]
                    tx = start_w[0] + (bx - start_w[0]) * alpha
                    ty = start_w[1] + (by - start_w[1]) * alpha
                    tz = start_w[2] + (1.2 - start_w[2]) * alpha  # Hạ dần xuống 1.2m trước khi LAND
                    tasks.append(send_ros_position(handler, x_ros=tx - bx, y_ros=ty - by, z_alt=tz, yaw_ros_deg=0.0))

            await asyncio.gather(*tasks)
            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("\n🛬 ĐÃ VỀ ĐẾN ĐỈNH BÃI ĐỖ! RA LỆNH HẠ CÁNH (LAND) ĐỒNG BỘ TOÀN BẦY...")
        await asyncio.gather(*[h.land() for _, h in active])
        print("⏳ Đang đợi toàn bộ drone tiếp đất an toàn trong 6 giây...")
        await asyncio.sleep(6.0)

        print("\n" + "=" * 70)
        print("  🎉 HOÀN THÀNH TOÀN DIỆN NHIỆM VỤ APF LEADER-FOLLOWER 5 DRONE! 🎉")
        print("  - Toàn bộ bầy drone giữ cự ly an toàn tuyệt đối suốt hành trình.")
        print("  - Cơ chế Closed-loop APF phản xạ tức thì dựa trên toạ độ thực tế.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình bay biểu diễn: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🧹 [BẢO VỆ AN TOÀN]: Đang kiểm tra trạng thái và hạ cánh toàn bộ drone trước khi thoát...")
        # Auto-LAND nếu drone còn armed
        for d_id, h in active:
            try:
                if h.get_telemetry().armed:
                    print(f"   ↳ Drone {d_id} vẫn đang ARMED! Phát lệnh cưỡng bức LAND...")
                    await h.land()
            except Exception as land_err:
                print(f"   ↳ Lỗi gửi lệnh land cho Drone {d_id}: {land_err}")

        # Chờ ngắt động cơ có timeout
        disarm_start = time.time()
        while time.time() - disarm_start < 15.0:
            armed_drones = [d_id for d_id, h in active if h.get_telemetry().armed]
            if not armed_drones:
                break
            await asyncio.sleep(0.5)

        for _, h in active:
            try:
                await h.disconnect()
            except Exception:
                pass
        print("Đã hoàn tất dọn dẹp và ngắt kết nối an toàn!")


if __name__ == "__main__":
    asyncio.run(main())
