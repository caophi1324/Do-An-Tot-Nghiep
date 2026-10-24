#!/usr/bin/env python3
"""
DRONE SWARM GAMEPAD CONTROLLER (TOÀN BỘ 5 DRONES)
Điều khiển 1 drone chỉ định hoặc ĐỒNG THỜI CẢ 5 DRONE bằng Tay Cầm (Xbox, PlayStation, Logitech, RC USB).
"""

import asyncio
import math
import os
import sys
import time

# Chống lỗi font console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm đường dẫn project root vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json

from core.mavlink_handler import MAVLinkHandler

DEADZONE = 0.12  # Vùng chết chống trôi mặc định


def normalize_axis(raw_val: float, axis_info: dict = None) -> float:
    """Chuẩn hóa giá trị cần gạt dựa theo điểm tâm nghỉ và biên độ thực tế."""
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


async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int) -> bool:
    """Kết nối tới ArduCopter SERIAL2 TCP (5773, 5783, 5793, 5803, 5813)."""
    connection_string = f"tcp:127.0.0.1:{port}"
    print(f"  [Drone {sysid}] Đang kết nối tới {connection_string}...")

    connected = await handler.connect(connection_string)
    if not connected:
        print(f"  [Drone {sysid}] ❌ Lỗi kết nối!")
        return False

    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    handler._mavconn.param_set_send("AVOID_ENABLE", 0)
    handler._mavconn.param_set_send("DISARM_DELAY", 0)
    await asyncio.sleep(0.3)
    print(f"  [Drone {sysid}] ✅ Đã kết nối sẵn sàng!")
    return True


async def main():
    print("=" * 65)
    print("   🎮 GAMEPAD ĐIỀU KHIỂN BẦY ĐÀN (SWARM CONTROLLER - 5 DRONES)")
    print("   Hỗ trợ: Xbox, PlayStation DualShock/DualSense, Logitech, RC USB")
    print("=" * 65 + "\n")

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

    print(f"\n🎮 ĐÃ NHẬN DIỆN TAY CẦM: [{js.get_name()}]")
    print(f"   - Số trục Analog: {js.get_numaxes()}")
    print(f"   - Số nút bấm    : {js.get_numbuttons()}\n")

    # Kết nối toàn bộ 5 Drone qua SERIAL2 (5773, 5783, 5793, 5803, 5813)
    num_drones = 5
    ports = [5773, 5783, 5793, 5803, 5813]
    handlers = [MAVLinkHandler() for _ in range(num_drones)]

    print(f"📡 Đang kết nối tới 5 drone qua cổng MAVLink {ports}...")
    setup_tasks = [setup_drone(handlers[i], ports[i], i + 1) for i in range(num_drones)]
    results = await asyncio.gather(*setup_tasks)

    active = [(i, handlers[i]) for i in range(num_drones) if results[i]]
    if not active:
        print("❌ Không có drone nào kết nối được! Hãy chắc chắn SITL đang chạy.")
        return

    print(f"\n✅ ĐÃ KẾT NỐI THÀNH CÔNG {len(active)}/5 DRONES!")

    print("\n" + "=" * 65)
    print("             📖 HƯỚNG DẪN ĐIỀU KHIỂN TAY CẦM BẦY ĐÀN")
    print("=" * 65)
    print("  🕹️ CẦN TRÁI (Left Stick)   : Lên/Xuống = Độ cao (Z) | Trái/Phải = Xoay đầu (Yaw)")
    print("  🕹️ CẦN PHẢI (Right Stick) : Lên/Xuống = Tiến/Lùi (X) | Trái/Phải = Lượn Ngang (Y)")
    print("  🔘 NÚT A / ❌ (Button 0)   : 🛫 Arm & Cất cánh lên 1.5m (Takeoff)")
    print("  🔘 NÚT B / ⭕ (Button 1)   : 🛬 Hạ cánh an toàn (LAND)")
    print("  🔘 NÚT LB / RB (L1 / R1)  : 🔄 Chuyển mục tiêu (Drone 1 ➔ 2 ➔ 3 ➔ 4 ➔ 5 ➔ TẤT CẢ)")
    print("  🔘 NÚT Y / 🔺 (Button 3)   : 🛑 Phanh dừng khẩn cấp / Hover tại chỗ")
    print("=" * 65 + "\n")

    # Target Drone Mode: -1 = CẢ 5 DRONE, 0..4 = Drone 1..5
    current_target = -1  # Mặc định điều khiển CẢ 5 DRONE CÙNG LÚC
    target_names = {
        -1: "⭐ TOÀN BỘ 5 DRONES (SWARM)",
        0: "🔴 Drone 1 (Leader)",
        1: "🟢 Drone 2",
        2: "🔵 Drone 3",
        3: "🟡 Drone 4",
        4: "🟣 Drone 5",
    }

    # Giới hạn vận tốc bay an toàn (m/s)
    MAX_SPEED_XY = 2.5   # 2.5 m/s
    MAX_SPEED_Z = 1.2    # 1.2 m/s
    MAX_YAW_RATE = 40.0  # 40 độ/s

    last_btn_time = 0.0

    profile_path = os.path.join(project_root, "config", "gamepad_profile.json")
    profile = None
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                print("📖 Đã tự động nạp cấu hình hiệu chuẩn: [config/gamepad_profile.json]")
        except Exception:
            pass
    axes_cfg = profile.get("axes", {}) if profile else {}

    try:
        while True:
            pygame.event.pump()
            now = time.time()

            # Đọc các trục Analog (Mode 2 Drone RC chuẩn với chuẩn hóa hiệu chuẩn)
            axis_yaw = normalize_axis(js.get_axis(0), axes_cfg.get("axis_0"))
            axis_climb = -normalize_axis(js.get_axis(1), axes_cfg.get("axis_1"))
            axis_roll = normalize_axis(js.get_axis(2) if js.get_numaxes() > 2 else 0.0, axes_cfg.get("axis_2"))
            axis_pitch = -normalize_axis(js.get_axis(3) if js.get_numaxes() > 3 else 0.0, axes_cfg.get("axis_3"))

            # Tính toán vận tốc (m/s)
            vx = axis_pitch * MAX_SPEED_XY     # Tiến (+) / Lùi (-)
            vy = axis_roll * MAX_SPEED_XY      # Phải (+) / Trái (-)
            vz = -axis_climb * MAX_SPEED_Z     # NED: Down (+) nên bay lên là -vz
            yaw_rate = axis_yaw * MAX_YAW_RATE # Xoay phải (+) / Xoay trái (-)

            # Đọc các nút bấm
            btn_a = js.get_button(0) if js.get_numbuttons() > 0 else 0
            btn_b = js.get_button(1) if js.get_numbuttons() > 1 else 0
            btn_y = js.get_button(3) if js.get_numbuttons() > 3 else 0
            btn_lb = js.get_button(4) if js.get_numbuttons() > 4 else 0
            btn_rb = js.get_button(5) if js.get_numbuttons() > 5 else 0
            btn_back = js.get_button(6) if js.get_numbuttons() > 6 else 0
            btn_start = js.get_button(7) if js.get_numbuttons() > 7 else 0

            # Nút BACK hoặc START: Thoát chương trình
            if btn_back or btn_start:
                print("\n\n👋 [NÚT BACK/START]: Đã nhận lệnh thoát chương trình từ tay cầm.")
                break

            # Xử lý nút LB / RB (Chuyển drone điều khiển: -1 -> 0 -> 1 -> 2 -> 3 -> 4 -> -1)
            if (btn_lb or btn_rb) and (now - last_btn_time > 0.3):
                last_btn_time = now
                if btn_rb:
                    current_target = (current_target + 2) % 6 - 1
                elif btn_lb:
                    current_target = (current_target if current_target != -1 else 5) - 1
                print(f"\n🎯 [CHUYỂN MỤC TIÊU]: Đang điều khiển 👉 {target_names[current_target]}")

            # Xử lý nút A: Takeoff 1.5m
            if btn_a and (now - last_btn_time > 1.0):
                last_btn_time = now
                print(f"\n🛫 [NÚT A]: Cất cánh lên 1.5m ({target_names[current_target]})...")
                targets = [h for i, h in active if (current_target == -1 or current_target == i)]
                await asyncio.gather(*[h.takeoff(altitude=1.5) for h in targets])

            # Xử lý nút B: LAND
            if btn_b and (now - last_btn_time > 1.0):
                last_btn_time = now
                print(f"\n🛬 [NÚT B]: Gửi lệnh HẠ CÁNH ({target_names[current_target]})...")
                targets = [h for i, h in active if (current_target == -1 or current_target == i)]
                await asyncio.gather(*[h.land() for h in targets])

            # Xử lý nút Y: Dừng khẩn cấp / Phanh lại
            if btn_y and (now - last_btn_time > 0.5):
                last_btn_time = now
                print(f"\n🛑 [NÚT Y]: Dừng và giữ vị trí tại chỗ!")
                targets = [h for i, h in active if (current_target == -1 or current_target == i)]
                await asyncio.gather(*[h.set_velocity(0, 0, 0, 0) for h in targets])

            # Gửi vận tốc điều khiển liên tục (nếu có gạt cần)
            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate) > 0.01:
                targets = [h for i, h in active if (current_target == -1 or current_target == i)]
                for h in targets:
                    await h.set_velocity(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

            # In trạng thái tốc độ
            target_str = target_names[current_target]
            print(
                f"\r🎮 [{target_str}] Vx: {vx:+.1f} m/s | Vy: {vy:+.1f} m/s | Vz: {-vz:+.1f} m/s | Yaw: {yaw_rate:+.0f}°/s    ",
                end="",
            )

            await asyncio.sleep(0.05)  # Tần số điều khiển 20Hz mượt mà

    except KeyboardInterrupt:
        print("\n\n✅ Đã nhận Ctrl+C. Dừng tay cầm điều khiển.")
    finally:
        for _, h in active:
            await h.disconnect()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
