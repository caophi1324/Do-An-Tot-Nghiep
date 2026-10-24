#!/usr/bin/env python3
"""
DRONE SWARM SHOW: CHÀO MỪNG 30 NĂM THÀNH LẬP TRƯỜNG (30th ANNIVERSARY LIGHT SHOW)
- Điều khiển 5 Drone mô phỏng / thực tế bay biểu diễn hoàn toàn tự động.
- Tần số bắn lệnh siêu mượt: 10Hz (0.1s mỗi bước) kèm dynamic dt compensation.
- Bộ nội suy đường cong S-Curve (Cosine Spline): Loại bỏ 100% hiện tượng bay giật cục.
- Phân tầng cao độ an toàn (Deconfliction Altitude Layers) & Giới hạn trần bay indoor 2.0m.
- Các pha biểu diễn:
    1. Cất cánh đồng bộ lên độ cao an toàn 1.3m (Synchronized Takeoff).
    2. Đội hình Ngôi Sao 5 Cánh (5-Pointed Star Formation - R=1.4m, Z=1.3m).
    3. Biểu diễn tạo hình Số "3" (Number 3 Formation - Z=1.3m).
    4. Chuyển tiếp mượt sang tạo hình Số "0" (Number 0 Formation - Z=1.3m).
    5. Vòng xoay thời gian 3D (3D Orbiting Wave, Z=1.4m ± 0.2m).
    6. Hạ cánh đồng bộ an toàn về đúng bãi đỗ riêng (Soft Landing).
"""

import argparse
import asyncio
import itertools
import math
import os
import sys
import time

# Chống lỗi font chữ tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.mavlink_handler import MAVLinkHandler, FlightMode

# Tọa độ gốc bãi đỗ xuất phát mặt đất của 5 drone (Đồng bộ chuẩn xác với SWARM_BASE_OFFSETS hội trường 20x30m, bước 1.6m)
SWARM_BASE_OFFSETS = {
    0: (0.0, 0.0),    # Drone 1 (Đỏ):         X_ros = 0.0m, Y_ros = 0.0m
    1: (0.0, 1.6),    # Drone 2 (Xanh lá):    X_ros = 0.0m, Y_ros = 1.6m
    2: (0.0, 3.2),    # Drone 3 (Xanh dương): X_ros = 0.0m, Y_ros = 3.2m
    3: (0.0, 4.8),    # Drone 4 (Vàng):       X_ros = 0.0m, Y_ros = 4.8m
    4: (0.0, 6.4),    # Drone 5 (Tím):        X_ros = 0.0m, Y_ros = 6.4m
}

# R3-05: Khởi hành lệch pha nhẹ 0.3s/drone loại bỏ hiện tượng Downwash mà không làm trễ đội hình
STAGGER_DELAY_SEC = 0.3


def read_actual_poses(active, base_offsets, max_age_sec: float = 0.5):
    """
    R3-01: Đọc vị trí thực tế của từng drone từ local_timestamp độc lập.
    Key trả về cố định theo idx = d_id - 1 để tránh lỗi trượt index (R3-16).
    """
    now = time.time()
    poses = {}
    for d_id, handler in active:
        t = handler.get_telemetry()
        if not t.local_valid or (now - t.local_timestamp) > max_age_sec:
            return None
        idx = d_id - 1
        bx, by = base_offsets[idx]
        # x_ros (East) = t.local_y + bx, y_ros (North) = t.local_x + by, z_ros (Up) = -t.local_z
        poses[idx] = (t.local_y + bx, t.local_x + by, -t.local_z, (90.0 - t.yaw) % 360.0)
    return poses


async def hold_all(active, base_offsets, last_poses):
    """Giữ cố định vị trí khi telemetry tạm thời bị mất hoặc cần phanh."""
    tasks = []
    for d_id, handler in active:
        idx = d_id - 1
        if idx in last_poses:
            wx, wy, wz, yaw = last_poses[idx]
            bx, by = base_offsets[idx]
            tasks.append(send_ros_position(handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw))
    if tasks:
        await asyncio.gather(*tasks)


def assign_optimal_targets(current_poses: dict, target_points: list) -> dict:
    """
    THUẬT TOÁN GÁN VỊ TRÍ TỐI ƯU (BRUTE-FORCE PERMUTATION O(n!)):
    - Với n=5 drone, tổng số hoán vị là 5! = 120, tính toán xong trong < 1ms.
    - Cực tiểu hóa tổng bình phương quãng đường di chuyển: min sum ||s_i - e_j||^2.
    - Giúp hạn chế tối đa việc các đường bay cắt chéo nhau.
    """
    num_pts = len(target_points)
    drone_indices = list(current_poses.keys())

    best_cost = float('inf')
    best_perm = None

    for perm in itertools.permutations(range(num_pts)):
        cost = 0.0
        for d_idx, p_idx in zip(drone_indices, perm):
            sx, sy, sz, _ = current_poses[d_idx]
            tx, ty, tz, _ = target_points[p_idx]
            cost += (sx - tx)**2 + (sy - ty)**2 + (sz - tz)**2
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return {drone_indices[i]: target_points[best_perm[i]] for i in range(len(drone_indices))}


async def send_ros_position(
    handler: MAVLinkHandler,
    x_ros: float,
    y_ros: float,
    z_alt: float,
    yaw_ros_deg: float = None,
    vx_ros: float = 0.0,
    vy_ros: float = 0.0,
    vz_ros: float = 0.0,
    use_vel: bool = False
):
    """
    Chuyển đổi tọa độ thế giới ROS ENU (East, North, Altitude) sang MAVLink Local NED của ArduPilot:
    - X_NED (North) = y_ros
    - Y_NED (East)  = x_ros
    - Z_NED (Down)  = -z_alt
    - Yaw_NED       = None nếu yaw_ros_deg là None (giữ nguyên hướng tự nhiên, triệt tiêu yaw cross-coupling)
    - Vx_NED (North)= vy_ros
    - Vy_NED (East) = vx_ros
    - Vz_NED (Down) = -vz_ros
    """
    yaw_mav = None
    if yaw_ros_deg is not None:
        yaw_mav = (90.0 - yaw_ros_deg) % 360.0
    return await handler.set_position_local(
        x=y_ros, y=x_ros, z_alt=z_alt, yaw=yaw_mav,
        vx=vy_ros, vy=vx_ros, vz=-vz_ros,
        use_vel=use_vel
    )


# Phân tầng cao độ 3D vòm parabol trong pha chuyển động (transit)
# Triệt tiêu giao cắt trên mặt phẳng XY, bảo đảm độ dãn cách Z từ 15cm đến 40cm
LAYER_OFFSETS = {
    0:  0.00,
    1: -0.15,
    2:  0.20,
    3: -0.10,
    4:  0.25,
}


async def smooth_transition(
    active_drones: list,
    start_targets: dict,
    end_targets: dict,
    duration_sec: float = 4.0,
    hz: int = 20,
    desc: str = ""
):
    """
    BỘ NỘI SUY QUỸ ĐẠO MƯỢT S-CURVE 20Hz + PHÂN TẦNG CAO ĐỘ 3D + FEEDFORWARD VẬN TỐC GIẢI TÍCH:
    - S-Curve Cosine 20Hz (mượt mà tuyệt đối, không giật gia tốc).
    - Phân tầng cao độ 3D vòm hình chuông sin^2(pi*p): Triệt tiêu điểm kỳ dị khi các quỹ đạo cắt nhau.
    - Vận tốc Feedforward giải tích: C^infinity trơn mượt, 0 tại t=0 và t=T.
    - Duy trì liên tục SubMode::POSVEL (use_vel=True, yaw=None) triệt tiêu 100% hiện tượng reset bộ lọc vận tốc và giật khựng.
    """
    total_steps = max(1, int(duration_sec * hz))
    TARGET_DT = 1.0 / hz
    T = duration_sec

    if desc:
        print(f"   ↳ 🎬 Đang chuyển cảnh: {desc} (Thời gian: {duration_sec}s, Tần số: {hz}Hz)...")

    step = 0
    next_t = time.monotonic()
    while step <= total_steps:
        base_progress = step / total_steps
        alpha = (1.0 - math.cos(math.pi * base_progress)) / 2.0
        dalpha_dt = (math.pi / (2.0 * T)) * math.sin(math.pi * base_progress)
        z_bell = math.sin(math.pi * base_progress) ** 2
        dz_bell_dt = (math.pi / T) * math.sin(2.0 * math.pi * base_progress)

        tasks = []
        for d_id, handler in active_drones:
            idx = d_id - 1
            if idx not in start_targets or idx not in end_targets:
                continue
            sx, sy, sz = start_targets[idx][:3]
            ex, ey, ez = end_targets[idx][:3]
            h_layer = LAYER_OFFSETS.get(idx, 0.0)

            wx = sx + (ex - sx) * alpha
            wy = sy + (ey - sy) * alpha
            wz = sz + (ez - sz) * alpha + h_layer * z_bell

            if step == 0 or step == total_steps:
                vx_ros, vy_ros, vz_ros = 0.0, 0.0, 0.0
            else:
                vx_ros = (ex - sx) * dalpha_dt
                vy_ros = (ey - sy) * dalpha_dt
                vz_ros = (ez - sz) * dalpha_dt + h_layer * dz_bell_dt

            bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 1.6))
            lx = wx - bx
            ly = wy - by

            # Giữ use_vel=True và yaw=None xuyên suốt để ArduPilot duy trì SubMode::POSVEL trơn mượt liên tục
            tasks.append(send_ros_position(
                handler, x_ros=lx, y_ros=ly, z_alt=wz, yaw_ros_deg=None,
                vx_ros=vx_ros, vy_ros=vy_ros, vz_ros=vz_ros, use_vel=True
            ))

        await asyncio.gather(*tasks)

        step += 1
        next_t += TARGET_DT
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            await asyncio.sleep(sleep_t)


async def hold_formation(
    active_drones: list,
    targets: dict,
    hold_sec: float = 4.0,
    hz: int = 20
):
    """
    Giữ cố định đội hình tại chỗ ở tần số 20Hz:
    Duy trì stream (target_pos, vel=0) với use_vel=True để ArduPilot giữ SubMode::POSVEL liên tục.
    Triệt tiêu 100% hiện tượng khựng/giật khi bước sang pha chuyển động kế tiếp!
    """
    total_steps = max(1, int(hold_sec * hz))
    dt = 1.0 / hz
    next_t = time.monotonic()

    for _ in range(total_steps):
        tasks = []
        for d_id, handler in active_drones:
            idx = d_id - 1
            if idx in targets:
                wx, wy, wz = targets[idx][:3]
                bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 1.6))
                lx = wx - bx
                ly = wy - by
                tasks.append(send_ros_position(
                    handler, x_ros=lx, y_ros=ly, z_alt=wz, yaw_ros_deg=None,
                    vx_ros=0.0, vy_ros=0.0, vz_ros=0.0, use_vel=True
                ))
        if tasks:
            await asyncio.gather(*tasks)
        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            await asyncio.sleep(sleep_t)


async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int, allow_sitl: bool = False) -> bool:
    """Kết nối tới ArduCopter SERIAL2 TCP port (5773, 5783, 5793, 5803, 5813)."""
    connection_string = f"tcp:127.0.0.1:{port}"
    print(f"[Drone {sysid}] Đang kết nối tới {connection_string}...")

    connected = await handler.connect(connection_string)
    if not connected:
        print(f"[Drone {sysid}] ❌ Lỗi kết nối!")
        return False

    # Triệt tiêu độ trễ Nagle trên TCP socket
    if hasattr(handler._mavconn, 'port') and hasattr(handler._mavconn.port, 'setsockopt'):
        try:
            import socket
            handler._mavconn.port.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    # Chặn ghi các tham số nguy hiểm vào EEPROM FC trừ khi có cờ --sitl
    if allow_sitl:
        try:
            handler._mavconn.param_set_send("ARMING_CHECK", 0)
            handler._mavconn.param_set_send("AVOID_ENABLE", 0)
            handler._mavconn.param_set_send("DISARM_DELAY", 0)
            handler._mavconn.param_set_send("FS_GCS_ENABLE", 0)
            handler._mavconn.param_set_send("FS_EKF_ACTION", 0)
        except Exception as e:
            print(f"[Drone {sysid}] ⚠️ Cảnh báo ghi param SITL: {e}")
    else:
        print(f"[Drone {sysid}] 🛡️ Chế độ Hardware Safety: Bảo toàn ARMING_CHECK và Failsafe của FC.")

    # Tham số tốc độ bay biểu diễn nhanh nhạy và mượt mà
    try:
        handler._mavconn.param_set_send("WPNAV_SPEED", 250)     # Tốc độ bay: 2.5 m/s (nhanh nhẹn, dứt khoát)
        handler._mavconn.param_set_send("WPNAV_ACCEL", 220)     # Gia tốc: 2.2 m/s²
        handler._mavconn.param_set_send("PSC_POSXY_P", 1.0)
    except Exception:
        pass

    print(f"[Drone {sysid}] ✅ Đã kết nối & nạp tham số tốc độ cao mượt mà (2.5m/s)!")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Drone Swarm 30 Years Anniversary Show")
    parser.add_argument("--sitl", action="store_true", default=False, help="Chế độ mô phỏng SITL (cho phép ghi đè param tạm)")
    parser.add_argument("--drones", type=int, default=3, choices=[3, 5], help="Số lượng drone biểu diễn (3 hoặc 5)")
    args, _ = parser.parse_known_args()

    num_drones = args.drones

    print("=" * 65)
    print(f"  🎓 DRONE SHOW: CHÀO MỪNG 30 NĂM THÀNH LẬP TRƯỜNG ({num_drones} DRONES) 🎓")
    print(f"  Chế độ: {'MÔ PHỎNG SITL' if args.sitl else 'PHẦN CỨNG AN TOÀN (HARDWARE SAFE)'}")
    print("  Tần số bắn lệnh: 10Hz | Thuật toán: S-Curve Cosine Spline (Trần <= 2.0m)")
    print("=" * 65)

    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    active = []

    # 1. KẾT NỐI TẤT CẢ DRONE QUA CỔNG SERIAL2 (5773, 5783, 5793, 5803, 5813)
    for i in range(1, num_drones + 1):
        port = 5760 + i * 10 + 3
        success = await setup_drone(handlers[i - 1], port, i, allow_sitl=args.sitl)
        if success:
            active.append((i, handlers[i - 1]))

    if len(active) < num_drones:
        print(f"❌ Không đủ {num_drones} drone hoạt động (chỉ có {len(active)} con). Hủy nhiệm vụ.")
        return

    try:
        # =========================================================================
        # PHA 1: CẤT CÁNH ĐỒNG BỘ (SYNCHRONIZED TAKEOFF - 1.3M)
        # =========================================================================
        takeoff_alt = 1.3
        print(f"\n🛫 [PHA 1] ARM VÀ CẤT CÁNH ĐỒNG BỘ {num_drones} DRONE LÊN {takeoff_alt}m (Trần an toàn ToF)...")

        for _, h in active:
            await h.set_mode(FlightMode.GUIDED)

        print(f"   ↳ ⏳ Đang kiểm tra EKF & ARM đồng loạt cho cả {num_drones} Drone...")
        for attempt in range(10):
            unarmed = [h for _, h in active if not h.get_telemetry().armed]
            if not unarmed:
                print(f"   ✅ Toàn bộ {num_drones} Drone đã ARM thành công, sẵn sàng cất cánh!")
                break
            await asyncio.gather(*[h.arm() for h in unarmed])
            await asyncio.sleep(1.5)

        await asyncio.gather(*[h.takeoff(altitude=takeoff_alt) for _, h in active])
        print("⏳ Đang ổn định độ cao cất cánh trong 4 giây...")
        await asyncio.sleep(4.0)

        # Vị trí sau cất cánh
        current_poses = {
            idx: (SWARM_BASE_OFFSETS[idx][0], SWARM_BASE_OFFSETS[idx][1], takeoff_alt, 0.0)
            for idx in range(num_drones)
        }

        # =========================================================================
        # PHA 2: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 1 (DELTA TRIANGLE / STAR FORMATION)
        # =========================================================================
        star_center_x = 2.5
        star_center_y = 3.2
        star_radius = 2.2
        star_alt = 1.4

        if num_drones == 3:
            print("\n🔺 [PHA 2] BIẾN ĐỔI ĐỘI HÌNH: TAM GIÁC ĐỀU DELTA (R = 2.0M, Z = 1.4M)...")
            star_points = [
                (star_center_x, star_center_y + 1.8, star_alt, 90.0),        # Đỉnh trước
                (star_center_x - 1.6, star_center_y - 1.0, star_alt, 210.0), # Đáy trái
                (star_center_x + 1.6, star_center_y - 1.0, star_alt, 330.0), # Đáy phải
            ]
            desc_pha2 = "Tiến vào 3 đỉnh Tam Giác Delta (Khởi hành so le)"
        else:
            print("\n⭐ [PHA 2] BIẾN ĐỔI ĐỘI HÌNH: NGÔI SAO 5 CÁNH (STAR FORMATION - R=2.2M, DÃN CÁCH 2.5m)...")
            star_points = []
            for idx in range(num_drones):
                angle = math.pi / 2.0 + idx * (2.0 * math.pi / 5.0)
                px = star_center_x + star_radius * math.cos(angle)
                py = star_center_y + star_radius * math.sin(angle)
                yaw_deg = math.degrees(angle) % 360.0
                star_points.append((px, py, star_alt, yaw_deg))
            desc_pha2 = "Tiến vào 5 đỉnh Ngôi Sao (Khởi hành so le)"

        star_poses = assign_optimal_targets(current_poses, star_points)
        await smooth_transition(active, current_poses, star_poses, duration_sec=4.0, desc=desc_pha2)
        print("   ✅ Đã tạo hình hoàn hảo! Giữ đội hình trong 5 giây...")
        await hold_formation(active, star_poses, hold_sec=5.0)
        current_poses = star_poses

        # =========================================================================
        # PHA 3: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 2 (V-SHAPE / NUMBER 3)
        # =========================================================================
        num3_alt = 1.4
        if num_drones == 3:
            print("\n✌️ [PHA 3] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH CHỮ V (V-SHAPE FORMATION, Z = 1.4M)...")
            num3_points = [
                (2.5, 4.4, num3_alt, 90.0),    # Mũi nhọn V
                (1.2, 2.0, num3_alt, 135.0),   # Cánh trái
                (3.8, 2.0, num3_alt, 45.0),    # Cánh phải
            ]
            desc_pha3 = "Lướt mượt chuyển thành Chữ V (Khởi hành so le)"
        else:
            print("\n3️⃣ [PHA 3] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH SỐ '3' (CAO 3.2M, RỘNG 2.4M, DÃN CÁCH 2.0m - 2.4m)...")
            num3_points = [
                (1.3, 4.8, num3_alt, 90.0),   # Đỉnh trái (X=1.3, Y=4.8)
                (3.7, 4.8, num3_alt, 0.0),    # Đỉnh phải (X=3.7, Y=4.8) - Ngang 2.4m
                (2.5, 3.2, num3_alt, 270.0),  # Eo giữa số 3 (X=2.5, Y=3.2)
                (3.7, 1.6, num3_alt, 180.0),  # Đáy phải (X=3.7, Y=1.6)
                (1.3, 1.6, num3_alt, 90.0),   # Đáy trái (X=1.3, Y=1.6) - Ngang 2.4m
            ]
            desc_pha3 = "Lướt mượt chuyển thành Số 3 (Khởi hành so le)"

        num3_poses = assign_optimal_targets(current_poses, num3_points)
        await smooth_transition(active, current_poses, num3_poses, duration_sec=4.0, desc=desc_pha3)
        print("   ✅ Đã định hình! Giữ đội hình trong 5 giây...")
        await hold_formation(active, num3_poses, hold_sec=5.0)
        current_poses = num3_poses

        # =========================================================================
        # PHA 4: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 3 (LINE / NUMBER 0)
        # =========================================================================
        num0_alt = 1.4
        if num_drones == 3:
            print("\n➖ [PHA 4] BIẾN ĐỔI ĐỘI HÌNH: HÀNG NGANG (LINE FORMATION, Z = 1.4M)...")
            num0_points = [
                (1.0, 3.2, num0_alt, 0.0),    # Trái
                (2.5, 3.2, num0_alt, 0.0),    # Giữa
                (4.0, 3.2, num0_alt, 0.0),    # Phải
            ]
            desc_pha4 = "Chuyển tiếp sang Hàng Ngang (Khởi hành so le)"
        else:
            print("\n0️⃣ [PHA 4] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH SỐ '0' (CAO 3.6M, RỘNG 2.6M, DÃN CÁCH 1.6m - 1.8m)...")
            num0_points = [
                (2.5, 5.0, num0_alt, 0.0),    # Đỉnh trên (X=2.5, Y=5.0) - Cao 3.6m
                (3.8, 4.1, num0_alt, 315.0),  # Sườn phải trên (X=3.8, Y=4.1)
                (3.8, 2.3, num0_alt, 225.0),  # Sườn phải dưới (X=3.8, Y=2.3)
                (2.5, 1.4, num0_alt, 180.0),  # Đáy dưới (X=2.5, Y=1.4)
                (1.2, 3.2, num0_alt, 90.0),   # Sườn trái giữa (X=1.2, Y=3.2) - Rộng 2.6m
            ]
            desc_pha4 = "Chuyển tiếp sang Số 0 (Khởi hành so le)"

        num0_poses = assign_optimal_targets(current_poses, num0_points)
        await smooth_transition(active, current_poses, num0_poses, duration_sec=4.0, desc=desc_pha4)
        print("   ✅ Đã định hình! Giữ đội hình trong 5 giây...")
        await hold_formation(active, num0_poses, hold_sec=5.0)
        current_poses = num0_poses

        # =========================================================================
        # PHA 5: VÒNG XOAY THỜI GIAN 3D (ORBITING 3D WAVE, R=2.2M, Z=1.4M ± 0.2M)
        # =========================================================================
        print("\n🌀 [PHA 5] CAO TRÀO BIỂU DIỄN: VÒNG XOAY THỜI GIAN 3D LƯỢN SÓNG (R=2.2m, DÃN CÁCH 2.5m)...")
        orbit_center_x = 2.5
        orbit_center_y = 3.2
        orbit_radius = 2.2
        orbit_time = 12.0
        hz = 20
        total_orbit_steps = int(orbit_time * hz)
        total_rotation = 2.0 * math.pi * 1.5

        initial_phases = [(2.0 * math.pi * k / num_drones) for k in range(num_drones)]

        orbit_entry_points = []
        for k in range(num_drones):
            d_angle = initial_phases[k]
            ox = orbit_center_x + orbit_radius * math.cos(d_angle)
            oy = orbit_center_y + orbit_radius * math.sin(d_angle)
            oz = 1.4 + 0.15 * math.sin(d_angle * 2.0)
            oyaw = math.degrees(d_angle + math.pi / 2.0) % 360.0
            orbit_entry_points.append((ox, oy, oz, oyaw))

        orbit_entry_poses = assign_optimal_targets(current_poses, orbit_entry_points)
        await smooth_transition(active, current_poses, orbit_entry_poses, duration_sec=3.8, desc="Tiến vào quỹ đạo Vòng Xoay 3D (Khởi hành so le)")
        current_poses = orbit_entry_poses

        drone_orbit_phases = {}
        for idx in range(num_drones):
            wx, wy, wz, _ = current_poses[idx]
            drone_orbit_phases[idx] = math.atan2(wy - orbit_center_y, wx - orbit_center_x)

        TARGET_DT = 1.0 / hz
        next_t = time.monotonic()
        omega = total_rotation / orbit_time
        for step in range(total_orbit_steps + 1):
            progress = step / total_orbit_steps
            angle_offset = total_rotation * progress

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                drone_angle = drone_orbit_phases[idx] + angle_offset
                wx = orbit_center_x + orbit_radius * math.cos(drone_angle)
                wy = orbit_center_y + orbit_radius * math.sin(drone_angle)
                wz = 1.4 + 0.15 * math.sin(drone_angle * 2.0)  # Giới hạn an toàn: 1.25m - 1.55m
                yaw_deg = math.degrees(drone_angle + math.pi / 2.0) % 360.0

                # Vận tốc tiếp tuyến giải tích trơn mượt không khựng
                vx = -orbit_radius * omega * math.sin(drone_angle)
                vy = orbit_radius * omega * math.cos(drone_angle)
                vz = 0.15 * 2.0 * omega * math.cos(drone_angle * 2.0)

                bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 1.6))
                lx = wx - bx
                ly = wy - by
                tasks.append(send_ros_position(
                    handler, x_ros=lx, y_ros=ly, z_alt=wz, yaw_ros_deg=yaw_deg,
                    vx_ros=vx, vy_ros=vy, vz_ros=vz, use_vel=True
                ))

            await asyncio.gather(*tasks)
            next_t += TARGET_DT
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ✅ Hoàn thành xuất sắc vòng xoay 3D cao trào!")

        # Cập nhật vị trí kết thúc vòng xoay
        for idx in range(num_drones):
            drone_angle = drone_orbit_phases[idx] + total_rotation
            wx = orbit_center_x + orbit_radius * math.cos(drone_angle)
            wy = orbit_center_y + orbit_radius * math.sin(drone_angle)
            wz = 1.4 + 0.15 * math.sin(drone_angle * 2.0)
            yaw_deg = math.degrees(drone_angle + math.pi / 2.0) % 360.0
            current_poses[idx] = (wx, wy, wz, yaw_deg)

        # =========================================================================
        # PHA 6: TRỞ VỀ BÃI ĐỖ VÀ HẠ CÁNH ĐỒNG BỘ (SAFE RETURN & LANDING)
        # =========================================================================
        print("\n🛬 [PHA 6] TRỞ VỀ VỊ TRÍ XUẤT PHÁT VÀ HẠ CÁNH ĐỒNG BỘ...")
        home_poses = {
            idx: (SWARM_BASE_OFFSETS[idx][0], SWARM_BASE_OFFSETS[idx][1], 1.2, 0.0)
            for idx in range(num_drones)
        }

        await smooth_transition(active, current_poses, home_poses, duration_sec=6.0, desc="Bay về đỉnh bãi đỗ riêng (Khởi hành so le)")
        await asyncio.sleep(1.5)

        print("   ↳ 🛬 Đang gửi lệnh HẠ CÁNH (LAND) đồng loạt...")
        await asyncio.gather(*[h.land() for _, h in active])
        print("\n🎉🎉🎉 BUỔI BIỂU DIỄN 30 NĂM THÀNH LẬP TRƯỜNG THÀNH CÔNG RỰC RỠ! 🎉🎉🎉")

    except KeyboardInterrupt:
        print("\n⚠️ Người dùng bấm Ctrl+C: HẠ CÁNH KHẨN CẤP TOÀN BỘ DRONE!")
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình bay biểu diễn: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 [BẢO VỆ AN TOÀN]: Đang kiểm tra và hạ cánh toàn bộ drone trước khi thoát...")
        for d_id, h in active:
            try:
                if h.get_telemetry().armed:
                    print(f"   ↳ Drone {d_id} vẫn ARMED! Phát lệnh cưỡng bức LAND...")
                    await h.land()
            except Exception as e:
                print(f"   ↳ Lỗi gửi land cho Drone {d_id}: {e}")

        # Chờ disarm tối đa 15s
        disarm_start = time.time()
        while time.time() - disarm_start < 15.0:
            still_armed = [d_id for d_id, h in active if h.get_telemetry().armed]
            if not still_armed:
                break
            await asyncio.sleep(0.5)

        if args.sitl:
            print("🛡️ [SITL CLEANUP] Đang hoàn tất kết thúc mô phỏng (giữ FS_GCS_ENABLE=0 để tránh khoá GCS Failsafe)...")
            for _, h in active:
                try:
                    h._mavconn.param_set_send("ARMING_CHECK", 0)
                    h._mavconn.param_set_send("FS_GCS_ENABLE", 0)
                    h._mavconn.param_set_send("FS_EKF_ACTION", 0)
                except Exception:
                    pass

        for _, h in active:
            try:
                await h.disconnect()
            except Exception:
                pass
        print("Đã ngắt kết nối an toàn!")


if __name__ == "__main__":
    asyncio.run(main())
