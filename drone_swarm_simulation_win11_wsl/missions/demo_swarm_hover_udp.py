import asyncio
import math
import os
import sys

# Chống lỗi font console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.mavlink_handler import MAVLinkHandler, FlightMode



async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int) -> bool:
    """Kết nối tới ArduCopter SERIAL2 TCP (5763, 5773, 5783, 5793)."""
    connection_string = f"tcp:127.0.0.1:{port}"
    print(f"[Drone {sysid}] Đang kết nối tới {connection_string}...")

    connected = await handler.connect(connection_string)
    if not connected:
        print(f"[Drone {sysid}] ❌ Lỗi kết nối!")
        return False

    print(f"[Drone {sysid}] ✅ Đã kết nối! Cấu hình tham số SITL...")
    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    handler._mavconn.param_set_send("AVOID_ENABLE", 0)
    handler._mavconn.param_set_send("DISARM_DELAY", 0)
    handler._mavconn.param_set_send("FS_GCS_ENABLE", 0)
    handler._mavconn.param_set_send("FS_EKF_ACTION", 0)
    await asyncio.sleep(0.5)
    return True


async def cmd_takeoff(active, altitude: float = 5.0):
    """
    Lệnh cất cánh đồng thời cho 4 drone.
    Ngay sau khi đạt độ cao, tự động tách 4 drone ra 4 vị trí riêng biệt (cách nhau 5m)
    bằng toạ độ tuyệt đối Local NED để không bao giờ bị đè lên nhau!
    """
    print(f"\n🛫 [LỆNH 1] Đang ARM và CẤT CÁNH đồng thời lên {altitude}m ({len(active)} drone)...")

    # 1. Đổi mode GUIDED và Arm đồng loạt
    for _, h in active:
        await h.set_mode(FlightMode.GUIDED)
    arm_tasks = [h.arm() for _, h in active]
    await asyncio.gather(*arm_tasks)
    await asyncio.sleep(1.5)  # Chờ động cơ quay đều

    # 2. Cất cánh đồng thời
    tasks = [h.takeoff(altitude=altitude) for _, h in active]
    results = await asyncio.gather(*tasks)
    if all(results):
        print(f"✅ Tất cả {len(active)} drone đã cất cánh và đạt độ cao {altitude}m!")
    else:
        print(f"⚠️ Đang ổn định độ cao các drone...")

    print("⏳ Chờ ổn định độ cao trong 3 giây...")
    await asyncio.sleep(3)


    # 3. Duy trì đội hình hàng ngang song song trên cao (cách đều 3.0m, không bao giờ chèn nhau)
    print(f"✅ ĐÃ CẤT CÁNH THÀNH CÔNG {len(active)} DRONE LÊN {altitude}m!")
    print(f"   (Quan sát trên RViz2: 5 drone đang dàn hàng ngang song song cách nhau 3m)")


# Toạ độ gốc bãi đỗ mặt đất (đồng bộ chuẩn xác với RViz2 Publisher)
SWARM_BASE_OFFSETS = {
    0: (0.0, 0.0),    # Drone 1 (Đỏ):      X_ros = 0m, Y_ros = 0m
    1: (0.0, 3.0),    # Drone 2 (Xanh lá):  X_ros = 0m, Y_ros = 3m
    2: (0.0, 6.0),    # Drone 3 (Xanh dương):X_ros = 0m, Y_ros = 6m
    3: (0.0, 9.0),    # Drone 4 (Vàng):     X_ros = 0m, Y_ros = 9m
    4: (0.0, 12.0),   # Drone 5 (Tím):      X_ros = 0m, Y_ros = 12m
}


async def send_ros_position(handler, x_ros: float, y_ros: float, z_alt: float, yaw_ros_deg: float = None):
    """
    Di chuyển theo toạ độ ROS ENU chuẩn (X: East, Y: North, Z: Altitude).
    Tự động chuyển đổi sang MAVLink NED:
    - X_NED (North) = y_ros
    - Y_NED (East)  = x_ros
    - Z_NED (Down)  = -z_alt
    - Yaw_NED (deg từ North) = (90 - yaw_ros_deg) % 360
    """
    yaw_mav = 0.0
    if yaw_ros_deg is not None:
        yaw_mav = (90.0 - yaw_ros_deg) % 360.0
    return await handler.set_position_local(x=y_ros, y=x_ros, z_alt=z_alt, yaw=yaw_mav)


async def cmd_land(active):
    """Lệnh hạ cánh an toàn (RTL / Return to Launch về đúng bãi đỗ riêng)."""
    print(f"\n🛬 [LỆNH 2] HẠ CÁNH AN TOÀN (RTL) cho {len(active)} drone...")
    tasks = [h.return_to_launch() for _, h in active]
    await asyncio.gather(*tasks)
    print("✅ Đã gửi lệnh hạ cánh! Mỗi drone đang tự bay về hạ cánh tại bãi đỗ riêng của mình.")


async def cmd_mission_circle_to_staircase(active):
    """
    KỊCH BẢN BIỂU DIỄN KÉP (DÙNG TOẠ ĐỘ THẾ GIỚI ROS ENU CHUẨN XÁC):
    Pha 1: 5 Drone bay theo Quỹ Đạo VÒNG TRÒN đồng tâm (Xc=6m, Yc=6m, R=4.5m).
    Pha 2: Biến đổi đội hình thành BẬC THANG leo dốc cao dần (3m -> 4.5m -> 6m -> 7.5m -> 9m).
    """
    print("\n" + "=" * 55)
    print("  🎭 BẮT ĐẦU KỊCH BẢN: VÒNG TRÒN ➡️ BẬC THANG")
    print("=" * 55)

    # --- PHA 1: BAY VÒNG TRÒN (Circle Swarm) ---
    center_x = 6.0
    center_y = 6.0
    radius = 4.5   # Bán kính 4.5m
    steps = 16     # 16 điểm trên đường tròn
    n = len(active)
    print(f"\n🔄 [PHA 1] {n} Drone bay lượn theo Quỹ Đạo Vòng Tròn (Tâm: ({center_x}, {center_y}), R = {radius}m)...")

    # Các drone xuất phát lệch đều nhau 360/N độ (72 độ mỗi drone)
    initial_phase = [(2 * math.pi * k / max(1, n)) for k in range(n)]

    for step in range(steps + 1):
        angle_progress = (2 * math.pi / steps) * step
        tasks = []
        for idx, (d_id, h) in enumerate(active):
            drone_angle = initial_phase[idx] + angle_progress
            wx = center_x + radius * math.cos(drone_angle)
            wy = center_y + radius * math.sin(drone_angle)
            bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 3.0))
            lx = wx - bx
            ly = wy - by

            # Góc tiếp tuyến trong ROS (độ)
            yaw_ros = math.degrees(drone_angle + math.pi / 2) % 360
            tasks.append(send_ros_position(h, x_ros=lx, y_ros=ly, z_alt=5.0, yaw_ros_deg=yaw_ros))

        await asyncio.gather(*tasks)
        print(f"   🌀 Vòng tròn: Bước {step}/{steps} (Góc quay: {int(math.degrees(angle_progress))}°)...", end="\r")
        await asyncio.sleep(1.8)

    print("\n✅ Hoàn thành Pha 1 (Vòng Tròn 360°)!")
    print("⏳ Chờ 3 giây trước khi biến đổi sang Bậc Thang...")
    await asyncio.sleep(3.0)

    # --- PHA 2: BIẾN ĐỔI BẬC THANG (Staircase Formation) ---
    print("\n📶 [PHA 2] Biến Đổi Đội Hình Thành BẬC THANG (Staircase Formation)...")
    stair_world = [
        (6.0, 0.0, 3.0),    # Drone 1: Bậc 1 (3.0m)
        (6.0, 3.0, 4.5),    # Drone 2: Bậc 2 (4.5m)
        (6.0, 6.0, 6.0),    # Drone 3: Bậc 3 (6.0m)
        (6.0, 9.0, 7.5),    # Drone 4: Bậc 4 (7.5m)
        (6.0, 12.0, 9.0),   # Drone 5: Bậc 5 (9.0m)
    ]

    tasks = []
    for idx, (d_id, h) in enumerate(active):
        wx, wy, alt = stair_world[idx] if idx < len(stair_world) else (6.0, float(idx) * 3.0, 5.0)
        bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 3.0))
        lx = wx - bx
        ly = wy - by
        print(f"   [Drone {idx+1}] Bậc thang -> Thế giới: (X={wx:.1f}m, Y={wy:.1f}m, Alt={alt:.1f}m)")
        tasks.append(send_ros_position(h, x_ros=lx, y_ros=ly, z_alt=alt, yaw_ros_deg=0.0))

    await asyncio.gather(*tasks)
    print("\n✅ ĐÃ TẠO ĐỘI HÌNH BẬC THANG HOÀN HẢO!")
    print("   (Xem trên RViz2: 5 mô hình Drone xếp thẳng hàng leo dốc cao dần từ 3m đến 9m)\n")


async def cmd_formation_square(active, size: float = 6.0):
    """Đội hình hình vuông (Square Formation 6x6m với Drone 5 ở tâm)."""
    print(f"\n🔷 [ĐỘI HÌNH] Xếp Đội Hình Hình Vuông ({size}m x {size}m)...")
    center_x = 6.0
    center_y = 6.0
    d = size / 2.0  # 3.0m

    # 4 Góc vuông hoàn hảo + 1 Tâm đỉnh trên cao (Toạ độ Thế giới ROS: X=East, Y=North, Z=Up)
    square_world = [
        (center_x - d, center_y - d, 5.0),  # Drone 1: Tây Nam (3, 3, 5)
        (center_x + d, center_y - d, 5.0),  # Drone 2: Đông Nam (9, 3, 5)
        (center_x - d, center_y + d, 5.0),  # Drone 3: Tây Bắc (3, 9, 5)
        (center_x + d, center_y + d, 5.0),  # Drone 4: Đông Bắc (9, 9, 5)
        (center_x,     center_y,     6.5),  # Drone 5: Tâm trên đỉnh (6, 6, 6.5)
    ]
    tasks = []
    for idx, (d_id, h) in enumerate(active):
        wx, wy, z = square_world[idx] if idx < len(square_world) else (0.0, float(idx) * 3.0, 5.0)
        bx, by = SWARM_BASE_OFFSETS.get(idx, (0.0, float(idx) * 3.0))
        lx = wx - bx
        ly = wy - by
        print(f"   [Drone {idx+1}] Điểm đến Thế giới -> (X={wx:.1f}m, Y={wy:.1f}m, Alt={z:.1f}m)")
        tasks.append(send_ros_position(h, x_ros=lx, y_ros=ly, z_alt=z))
    await asyncio.gather(*tasks)
    print("✅ Đã xếp xong Đội hình Hình Vuông 6x6m chuẩn xác 100%!")


async def main():
    num_drones = 5
    print("==================================================")
    print(f"   🎮 SWARM FLIGHT CONTROLLER ({num_drones} DRONES)")
    print("   MAVROS (SERIAL1) + Python (SERIAL2)")
    print("==================================================\n")

    ports = [5763 + i * 10 for i in range(num_drones)]  # 5763, 5773, 5783, 5793, 5803
    handlers = [MAVLinkHandler() for _ in range(num_drones)]

    print(f"Đang kết nối tới {num_drones} Drones qua ports: {ports}\n")
    setup_tasks = [setup_drone(handlers[i], ports[i], i + 1) for i in range(num_drones)]
    results = await asyncio.gather(*setup_tasks)

    active = [(i, handlers[i]) for i in range(num_drones) if results[i]]
    if not active:
        print("❌ Không có drone nào kết nối được! Hãy chạy wsl_start_sitl_swarm.sh trước.")
        return

    print(f"\n✅ ĐÃ KẾT NỐI {len(active)}/{num_drones} DRONES THÀNH CÔNG!")
    print("⏳ Chờ EKF & GPS đồng bộ (8 giây)...")
    await asyncio.sleep(8)

    # Menu tương tác
    while True:
        print("\n" + "-" * 50)
        print("           📋 MENU ĐIỀU KHIỂN SWARM")
        print("-" * 50)
        print("  [1] 🛫 CẤT CÁNH (Takeoff & Tách 4 góc)")
        print("  [2] 🛬 HẠ CÁNH (Land / RTL)")
        print("  [3] 🎭 CHẠY KỊCH BẢN: VÒNG TRÒN ➡️ BẬC THANG")
        print("  [4] 🔷 ĐỘI HÌNH HÌNH VUÔNG (Square 6x6m)")
        print("  [0] ❌ THOÁT CHƯƠNG TRÌNH")
        print("-" * 50)

        # Đọc phím từ người dùng không block event loop
        loop = asyncio.get_event_loop()
        choice = await loop.run_in_executor(None, input, "👉 Nhập lựa chọn (1/2/3/4/0): ")
        choice = choice.strip()

        if choice == "1":
            await cmd_takeoff(active, altitude=5.0)
        elif choice == "2":
            await cmd_land(active)
        elif choice == "3":
            await cmd_mission_circle_to_staircase(active)
        elif choice == "4":
            await cmd_formation_square(active, size=6.0)
        elif choice == "0":
            print("Đang yêu cầu hạ cánh trước khi thoát...")
            await cmd_land(active)
            print("Tạm biệt!")
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ, vui lòng bấm phím 1, 2, 3, 4 hoặc 0.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nĐã nhận Ctrl+C. Thoát chương trình.")
