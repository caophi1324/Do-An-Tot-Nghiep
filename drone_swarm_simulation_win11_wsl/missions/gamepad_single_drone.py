#!/usr/bin/env python3
"""
GAMEPAD CONTROLLER CHO 1 DRONE ĐỘC LẬP (DRONE 1)
Hỗ trợ: Xbox (One/Series/360), PlayStation (DualShock 4/DualSense), Logitech, Flysky/Radiomaster USB & Generic Gamepad.
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


async def main():
    print("=" * 65)
    print("   🎮 TAY CẦM ĐIỀU KHIỂN 1 DRONE (SINGLE DRONE PILOT - DRONE 1)")
    print("   Hỗ trợ: Xbox, PS4/PS5, Logitech, Flysky/Radiomaster & USB Gamepad")
    print("=" * 65 + "\n")

    try:
        import pygame
    except ImportError:
        print("❌ Chưa cài đặt thư viện 'pygame'!")
        print("👉 Hãy chạy lệnh: python -m pip install pygame")
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
    print(f"   - Số nút bấm    : {js.get_numbuttons()}")

    # Kết nối Drone 1 qua cổng SERIAL2 MAVLink (TCP 5773)
    port = 5773
    handler = MAVLinkHandler()
    conn_str = f"tcp:127.0.0.1:{port}"
    print(f"\n📡 Đang kết nối tới Drone 1 ({conn_str})...")

    connected = await handler.connect(conn_str)
    if not connected:
        print("❌ Lỗi kết nối tới Drone 1! Hãy chắc chắn SITL đang chạy.")
        return

    # Gỡ bỏ kiểm tra an toàn trong SITL
    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    handler._mavconn.param_set_send("AVOID_ENABLE", 0)
    handler._mavconn.param_set_send("DISARM_DELAY", 0)
    await asyncio.sleep(0.5)

    print("\n✅ ĐÃ SẴN SÀNG ĐIỀU KHIỂN DRONE 1!")
    print("=" * 65)
    print("  🔘 NÚT A / ❌   : 🛫 Arm & Cất cánh lên 1.5m (Takeoff)")
    print("  🔘 NÚT B / ⭕   : 🛬 Hạ cánh an toàn (LAND thẳng đứng)")
    print("  🔘 NÚT Y / 🔺   : 🛑 Phanh khẩn cấp / Giữ vị trí tại chỗ")
    print("  🕹️ CẦN TRÁI    : Lên/Xuống = Độ cao (Z) | Trái/Phải = Xoay đầu (Yaw)")
    print("  🕹️ CẦN PHẢI    : Lên/Xuống = Tiến/Lùi (X) | Trái/Phải = Bay ngang (Y)")
    print("=" * 65 + "\n")

    MAX_SPEED_XY = 2.5    # Tốc độ tiến lùi ngang tối đa 2.5 m/s (an toàn trong nhà)
    MAX_SPEED_Z = 1.2     # Tốc độ nâng hạ độ cao tối đa 1.2 m/s
    MAX_YAW_RATE = 45.0   # Tốc độ xoay đầu tối đa 45 độ/s

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

            # Đọc các trục Analog (Mode 2 Drone RC chuẩn quốc tế với chuẩn hóa hiệu chuẩn)
            axis_yaw = normalize_axis(js.get_axis(0), axes_cfg.get("axis_0"))
            axis_climb = -normalize_axis(js.get_axis(1), axes_cfg.get("axis_1"))
            axis_roll = normalize_axis(js.get_axis(2) if js.get_numaxes() > 2 else 0.0, axes_cfg.get("axis_2"))
            axis_pitch = -normalize_axis(js.get_axis(3) if js.get_numaxes() > 3 else 0.0, axes_cfg.get("axis_3"))

            # Tính vận tốc (m/s)
            vx = axis_pitch * MAX_SPEED_XY
            vy = axis_roll * MAX_SPEED_XY
            vz = -axis_climb * MAX_SPEED_Z
            yaw_rate = axis_yaw * MAX_YAW_RATE

            # Nút bấm
            btn_a = js.get_button(0) if js.get_numbuttons() > 0 else 0
            btn_b = js.get_button(1) if js.get_numbuttons() > 1 else 0
            btn_y = js.get_button(3) if js.get_numbuttons() > 3 else 0
            btn_back = js.get_button(6) if js.get_numbuttons() > 6 else 0
            btn_start = js.get_button(7) if js.get_numbuttons() > 7 else 0

            # Nút BACK hoặc START: Thoát chương trình
            if btn_back or btn_start:
                print("\n\n👋 [NÚT BACK/START]: Đã nhận lệnh thoát chương trình từ tay cầm.")
                break

            # Xử lý Nút A: Takeoff 1.5m
            if btn_a and (now - last_btn_time > 1.2):
                last_btn_time = now
                print(f"\n🛫 [NÚT A]: Đang Arm và Cất cánh lên 1.5m...")
                await handler.takeoff(altitude=1.5)

            # Xử lý Nút B: LAND
            if btn_b and (now - last_btn_time > 1.2):
                last_btn_time = now
                print(f"\n🛬 [NÚT B]: Đang gửi lệnh HẠ CÁNH (LAND)...")
                await handler.land()

            # Xử lý Nút Y: Phanh khẩn cấp
            if btn_y and (now - last_btn_time > 0.5):
                last_btn_time = now
                print(f"\n🛑 [NÚT Y]: Đang phanh khẩn cấp & giữ vị trí!")
                await handler.set_velocity(0.0, 0.0, 0.0, 0.0)

            # Gửi lệnh vận tốc liên tục nếu có gạt cần
            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate) > 0.01:
                await handler.set_velocity(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

            # Live HUD hiển thị trên 1 dòng duy nhất
            telem = handler.telemetry
            print(
                f"\r🚁 Alt: {telem.altitude:.2f}m | Mode: {telem.flight_mode} | "
                f"Vx: {vx:+.1f} m/s | Vy: {vy:+.1f} m/s | Vz: {-vz:+.1f} m/s | Yaw: {yaw_rate:+.0f}°/s | [Bấm BACK/START hoặc Ctrl+C để thoát]   ",
                end="",
            )

            await asyncio.sleep(0.05)  # 20Hz mượt mà

    except KeyboardInterrupt:
        print("\n\n✅ Đã nhận Ctrl+C. Dừng tay cầm điều khiển.")
    finally:
        await handler.disconnect()



if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
