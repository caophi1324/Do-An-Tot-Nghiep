#!/usr/bin/env python3
"""
DRONE SHOW 30 NĂM THÀNH LẬP TRƯỜNG - CÔNG NGHỆ ORCA / RVO2 + HUNGARIAN + MINIMUM SNAP
- Độc lập 100% khỏi APF (sử dụng package orca_swarm)
- Gán đích tối ưu Hungarian: min sum ||p_i - g_pi(i)||^2, triệt tiêu 100% cắt chéo đường bay
- Quỹ đạo Minimum Snap đa thức bậc 7: Triệt tiêu gia tốc và jerk ở 2 đầu mút
- Lõi né tránh va chạm ORCA/RVO2: Tự do điều hướng trong không gian vận tốc
- Quy mô hình học chuẩn hội trường 20x30m (Cự ly dãn cách 1.6m - 2.6m)
- 6 Pha hoàn chỉnh:
  1. Cất cánh đồng bộ 1.3m
  2. Tạo hình Ngôi sao 5 cánh (R = 2.2m)
  3. Tạo hình Số "3" (Cao 3.2m, Rộng 2.4m)
  4. Tạo hình Số "0" (Cao 3.6m, Rộng 2.6m)
  5. Vòng xoay thời gian 3D lượn sóng (R = 2.2m, z = 1.4m ± 0.2m)
  6. Bay về đỉnh bãi đỗ riêng và hạ cánh đồng bộ (LAND)
"""

import argparse
import asyncio
import math
import os
import sys
import time

# Chống lỗi font chữ tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import core và orca_swarm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.mavlink_handler import MAVLinkHandler, FlightMode
from orca_swarm import (
    ORCASwarmController,
    ORCAConfig,
    HungarianAssigner,
    MinimumSnapTrajectory,
    BSplineTrajectory,
)
from orca_swarm.orca_controller import SWARM_BASE_OFFSETS, send_ros_position


async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int, allow_sitl: bool = False) -> bool:
    """Kết nối MAVLink tới cổng TCP và nạp tham số ban đầu."""
    conn_str = f"tcp:127.0.0.1:{port}"
    print(f"[Drone {sysid}] Đang kết nối tới {conn_str}...")

    connected = await handler.connect(conn_str)
    if not connected:
        print(f"[Drone {sysid}] ❌ Lỗi kết nối!")
        return False

    if allow_sitl:
        try:
            handler._mavconn.param_set_send("ARMING_CHECK", 0)
            handler._mavconn.param_set_send("AVOID_ENABLE", 0)
            handler._mavconn.param_set_send("DISARM_DELAY", 0)
            handler._mavconn.param_set_send("FS_GCS_ENABLE", 0)
            handler._mavconn.param_set_send("FS_EKF_ACTION", 0)
            handler._mavconn.param_set_send("FS_CRASH_CHECK", 0)
            handler._mavconn.param_set_send("BATT_LOW_MAH", 0)
            handler._mavconn.param_set_send("BATT_CRT_MAH", 0)
            handler._mavconn.param_set_send("BATT_FS_LOW_ACT", 0)
            handler._mavconn.param_set_send("BATT_FS_CRT_ACT", 0)
            handler._mavconn.param_set_send("SIM_BATT_CAP_AH", 0.0)
            handler._mavconn.param_set_send("SIM_BATT_VOLTAGE", 12.6)
        except Exception as e:
            print(f"[Drone {sysid}] ⚠️ Cảnh báo nạp param SITL: {e}")
    else:
        print(f"[Drone {sysid}] 🛡️ Chế độ Hardware Safe: Bảo toàn Failsafe của FC.")

    # Cấu hình vận tốc và gia tốc bám đuổi nhanh nhạy
    try:
        handler._mavconn.param_set_send("WPNAV_SPEED", 180)     # 1.8 m/s
        handler._mavconn.param_set_send("WPNAV_ACCEL", 150)     # 1.5 m/s²
    except Exception:
        pass

    print(f"[Drone {sysid}] ✅ Đã kết nối & nạp tham số ORCA mượt mà (1.8m/s)!")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Drone Swarm 30 Years Anniversary - ORCA / RVO2 Engine")
    parser.add_argument("--sitl", action="store_true", default=False, help="Chế độ mô phỏng SITL")
    parser.add_argument("--traj", type=str, default="min_snap", choices=["min_snap", "bspline"], help="Loại quỹ đạo (min_snap hoặc bspline)")
    parser.add_argument("--drones", type=int, default=3, choices=[3, 5], help="Số lượng drone biểu diễn (3 hoặc 5)")
    args, _ = parser.parse_known_args()

    num_drones = args.drones

    print("=" * 70)
    print(f"  🎓 DRONE SHOW: CHÀO MỪNG 30 NĂM THÀNH LẬP TRƯỜNG ({num_drones} DRONES) 🎓")
    print("  🚀 CÔNG NGHỆ: ORCA/RVO2 + HUNGARIAN + MINIMUM SNAP (orca_swarm)")
    print(f"  Chế độ: {'MÔ PHỎNG SITL' if args.sitl else 'PHẦN CỨNG AN TOÀN'}")
    print(f"  Bộ tạo đường bay: {args.traj.upper()} | Không gian vận tốc tự do")
    print("=" * 70)

    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    active = []

    # 1. Kết nối các drone qua cổng SERIAL2
    for i in range(1, num_drones + 1):
        port = 5760 + i * 10 + 3
        success = await setup_drone(handlers[i - 1], port, i, allow_sitl=args.sitl)
        if success:
            active.append((i, handlers[i - 1]))

    if len(active) < num_drones:
        print(f"❌ Không đủ {num_drones} drone hoạt động (chỉ có {len(active)} con). Hủy nhiệm vụ.")
        return

    # Khởi tạo Bộ điều phối ORCA Swarm Controller
    orca_cfg = ORCAConfig(
        time_horizon=2.5,
        agent_radius=0.45,
        max_speed=1.6,
        time_step=0.05,
        safety_margin=0.25
    )
    controller = ORCASwarmController(orca_cfg, base_offsets=SWARM_BASE_OFFSETS, use_3d=False)

    try:
        # =========================================================================
        # PHA 1: CẤT CÁNH ĐỒNG BỘ (SYNCHRONIZED TAKEOFF - 2.2M)
        # =========================================================================
        takeoff_alt = 2.2
        print(f"\n🛫 [PHA 1] ARM VÀ CẤT CÁNH ĐỒNG BỘ {num_drones} DRONE LÊN {takeoff_alt}m (Độ cao thoáng đãng, triệt tiêu hiệu ứng mặt đất)...")

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

        # =========================================================================
        # PHA 2: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 1 (DELTA TRIANGLE / STAR FORMATION)
        # =========================================================================
        star_cx, star_cy = 2.5, 3.2
        star_r = 2.2
        star_alt = 2.2

        if num_drones == 3:
            print("\n🔺 [PHA 2] BIẾN ĐỔI ĐỘI HÌNH: TAM GIÁC ĐỀU DELTA (R = 2.0M, Z = 2.2M)...")
            star_points = [
                (star_cx, star_cy + 1.8, star_alt, 90.0),        # Đỉnh trước (mũi nhọn)
                (star_cx - 1.6, star_cy - 1.0, star_alt, 210.0), # Đáy trái
                (star_cx + 1.6, star_cy - 1.0, star_alt, 330.0), # Đáy phải
            ]
            desc_pha2 = "Tiến vào 3 đỉnh Tam Giác Delta (Hungarian + ORCA)"
        else:
            print("\n⭐ [PHA 2] BIẾN ĐỔI ĐỘI HÌNH: NGÔI SAO 5 CÁNH (R = 2.2M, Z = 2.2M, DÃN CÁCH 2.58M)...")
            star_points = []
            for idx in range(num_drones):
                angle = math.pi / 2.0 + idx * (2.0 * math.pi / 5.0)
                px = star_cx + star_r * math.cos(angle)
                py = star_cy + star_r * math.sin(angle)
                yaw_deg = math.degrees(angle) % 360.0
                star_points.append((px, py, star_alt, yaw_deg))
            desc_pha2 = "Tiến vào 5 đỉnh Ngôi Sao (Hungarian + ORCA)"

        star_targets = await controller.execute_transition(
            active, star_points, duration_sec=4.5, hz=20, traj_type=args.traj, desc=desc_pha2
        )
        print("   ✅ ĐÃ TẠO HÌNH XUẤT SẮC! Giữ đội hình 5 giây...")
        await controller.hold_formation(active, star_targets, hold_sec=5.0, hz=20)

        # =========================================================================
        # PHA 3: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 2 (V-SHAPE / NUMBER 3)
        # =========================================================================
        num3_alt = 2.2
        if num_drones == 3:
            print("\n✌️ [PHA 3] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH CHỮ V (V-SHAPE FORMATION, Z = 2.2M)...")
            num3_points = [
                (2.5, 4.4, num3_alt, 90.0),    # Mũi nhọn V
                (1.2, 2.0, num3_alt, 135.0),   # Cánh trái
                (3.8, 2.0, num3_alt, 45.0),    # Cánh phải
            ]
            desc_pha3 = "Chuyển mượt thành Chữ V (Hungarian + ORCA)"
        else:
            print("\n3️⃣ [PHA 3] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH SỐ '3' (CAO 3.2M, RỘNG 2.4M, Z = 2.2M)...")
            num3_points = [
                (1.3, 4.8, num3_alt, 90.0),   # Đỉnh trái
                (3.7, 4.8, num3_alt, 0.0),    # Đỉnh phải
                (2.5, 3.2, num3_alt, 270.0),  # Eo giữa số 3
                (3.7, 1.6, num3_alt, 180.0),  # Đáy phải
                (1.3, 1.6, num3_alt, 90.0),   # Đáy trái
            ]
            desc_pha3 = "Chuyển mượt thành Số 3 (Hungarian + ORCA)"

        num3_targets = await controller.execute_transition(
            active, num3_points, duration_sec=4.5, hz=20, traj_type=args.traj, desc=desc_pha3
        )
        print("   ✅ ĐÃ ĐỊNH HÌNH! Giữ đội hình 5 giây...")
        await controller.hold_formation(active, num3_targets, hold_sec=5.0, hz=20)

        # =========================================================================
        # PHA 4: BIẾN ĐỔI ĐỘI HÌNH ĐỢT 3 (LINE / NUMBER 0)
        # =========================================================================
        num0_alt = 2.2
        if num_drones == 3:
            print("\n➖ [PHA 4] BIẾN ĐỔI ĐỘI HÌNH: HÀNG NGANG (LINE FORMATION, Z = 2.2M)...")
            num0_points = [
                (1.0, 3.2, num0_alt, 0.0),    # Trái
                (2.5, 3.2, num0_alt, 0.0),    # Giữa
                (4.0, 3.2, num0_alt, 0.0),    # Phải
            ]
            desc_pha4 = "Chuyển tiếp sang Hàng Ngang (Hungarian + ORCA)"
        else:
            print("\n0️⃣ [PHA 4] BIẾN ĐỔI ĐỘI HÌNH: TẠO HÌNH SỐ '0' (CAO 3.6M, RỘNG 2.6M, Z = 2.2M)...")
            num0_points = [
                (2.5, 5.0, num0_alt, 0.0),    # Đỉnh trên
                (3.8, 4.1, num0_alt, 315.0),  # Sườn phải trên
                (3.8, 2.3, num0_alt, 225.0),  # Sườn phải dưới
                (2.5, 1.4, num0_alt, 180.0),  # Đáy dưới
                (1.2, 3.2, num0_alt, 90.0),   # Sườn trái giữa
            ]
            desc_pha4 = "Chuyển tiếp sang Số 0 (Hungarian + ORCA)"

        num0_targets = await controller.execute_transition(
            active, num0_points, duration_sec=4.5, hz=20, traj_type=args.traj, desc=desc_pha4
        )
        print("   ✅ ĐÃ ĐỊNH HÌNH! Giữ đội hình 5 giây...")
        await controller.hold_formation(active, num0_targets, hold_sec=5.0, hz=20)

        # =========================================================================
        # PHA 5: VÒNG XOAY THỜI GIAN 3D (3D ORBIT WAVE - R = 2.2M, Z = 2.2M ± 0.3M)
        # =========================================================================
        print("\n🌀 [PHA 5] CAO TRÀO BIỂU DIỄN: VÒNG XOAY THỜI GIAN 3D LƯỢN SÓNG (R = 2.2M, Z = 2.2M ± 0.3M)...")
        orbit_cx, orbit_cy = 2.5, 3.2
        orbit_r = 2.2
        orbit_time = 12.0
        hz = 20
        total_orbit_steps = int(orbit_time * hz)
        total_rotation = 2.0 * math.pi * 1.5

        # Điểm nhập quỹ đạo vòng xoay
        orbit_entry_points = []
        for k in range(num_drones):
            d_angle = 2.0 * math.pi * k / num_drones
            ox = orbit_cx + orbit_r * math.cos(d_angle)
            oy = orbit_cy + orbit_r * math.sin(d_angle)
            oz = 2.2 + 0.30 * math.sin(d_angle * 2.0)
            oyaw = math.degrees(d_angle + math.pi / 2.0) % 360.0
            orbit_entry_points.append((ox, oy, oz, oyaw))

        orbit_targets = await controller.execute_transition(
            active, orbit_entry_points, duration_sec=4.5, hz=20, traj_type=args.traj, desc="Nhập vào Vòng Xoay 3D (Hungarian + ORCA)"
        )

        # Xác định góc xuất phát chính xác cho từng drone từ vị trí Hungarian đã gán
        # Triệt tiêu hoàn toàn lỗi hoán đổi vị trí qua tâm vòng tròn (nguyên nhân gây Breach)
        drone_base_angles = {}
        for d_id, _ in active:
            idx = d_id - 1
            if idx in orbit_targets:
                tx, ty = orbit_targets[idx][0], orbit_targets[idx][1]
                drone_base_angles[idx] = math.atan2(ty - orbit_cy, tx - orbit_cx)
            else:
                drone_base_angles[idx] = 2.0 * math.pi * idx / num_drones

        # Vòng xoay động học 3D mượt mà, bảo toàn cự ly dãn cách 2.58m
        dt = 1.0 / hz
        next_t = time.monotonic()
        omega = total_rotation / (total_orbit_steps * dt)
        for step in range(total_orbit_steps + 1):
            progress = step / total_orbit_steps
            angle_offset = total_rotation * progress

            tasks = []
            for d_id, handler in active:
                idx = d_id - 1
                base_angle = drone_base_angles[idx]
                drone_angle = base_angle + angle_offset

                wx = orbit_cx + orbit_r * math.cos(drone_angle)
                wy = orbit_cy + orbit_r * math.sin(drone_angle)
                wz = 2.2 + 0.30 * math.sin(drone_angle * 2.0)
                yaw_deg = math.degrees(drone_angle + math.pi / 2.0) % 360.0

                # Vận tốc tiếp tuyến giải tích trơn mượt không khựng
                vx = -orbit_r * omega * math.sin(drone_angle)
                vy = orbit_r * omega * math.cos(drone_angle)
                vz = 0.30 * 2.0 * omega * math.cos(drone_angle * 2.0)

                bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 1.6))
                tasks.append(send_ros_position(
                    handler, x_ros=wx - bx, y_ros=wy - by, z_alt=wz, yaw_ros_deg=yaw_deg,
                    vx_ros=vx, vy_ros=vy, vz_ros=vz, use_vel=True
                ))

            await asyncio.gather(*tasks)
            next_t += dt
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print("   ✅ Hoàn thành xuất sắc vòng xoay 3D cao trào!")

        # =========================================================================
        # PHA 6: TRỞ VỀ BÃI ĐỖ VÀ HẠ CÁNH ĐỒNG BỘ
        # =========================================================================
        print("\n🛬 [PHA 6] TRỞ VỀ VỊ TRÍ XUẤT PHÁT VÀ HẠ CÁNH ĐỒNG BỘ...")
        home_points = [
            (SWARM_BASE_OFFSETS[i][0], SWARM_BASE_OFFSETS[i][1], 2.0, 0.0)
            for i in range(num_drones)
        ]

        await controller.execute_transition(
            active, home_points, duration_sec=6.0, hz=20, traj_type=args.traj, desc="Bay về bãi đỗ riêng (Hungarian + ORCA)"
        )
        await asyncio.sleep(1.0)

        print("   ↳ 🛬 Đang gửi lệnh HẠ CÁNH (LAND) đồng loạt...")
        await asyncio.gather(*[h.land() for _, h in active])
        print("\n🎉🎉🎉 BUỔI BIỂU DIỄN 30 NĂM BẰNG CÔNG NGHỆ ORCA THÀNH CÔNG RỰC RỠ! 🎉🎉🎉")

    except KeyboardInterrupt:
        print("\n⚠️ Người dùng bấm Ctrl+C: HẠ CÁNH KHẨN CẤP!")
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

        # Chờ disarm
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
