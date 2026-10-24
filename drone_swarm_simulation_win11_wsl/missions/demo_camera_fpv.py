"""
DRONEEXE — Demo: FPV Camera + Telemetry Overlay

Nhận stream MJPEG từ ESP32-CAM và hiển thị kèm telemetry realtime.
Có thể chạy song song với MAVLink handler hoặc standalone.

Chạy:
    # Chỉ xem camera (không cần drone kết nối):
    python scripts/demo_camera_fpv.py --camera https://192.168.4.2:81/stream

    # Camera + MAVLink telemetry overlay:
    python scripts/demo_camera_fpv.py --camera https://192.168.4.2:81/stream --connection udp:127.0.0.1:14550

    # HTTPS (ESP32 với self-signed cert):
    python scripts/demo_camera_fpv.py --camera https://192.168.4.2/stream

Yêu cầu:
    pip install opencv-python requests pymavlink loguru
"""

import asyncio
import argparse
import sys
import os
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera_handler import CameraHandler
from core.mavlink_handler import MAVLinkHandler
from core.base_handler import DroneType, TelemetryData


# ── Telemetry overlay ────────────────────────────────────────────────────────

def build_overlay(frame: np.ndarray, telemetry: Optional[TelemetryData], fps: float) -> np.ndarray:
    """Vẽ thông tin telemetry và FPS lên frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Nền mờ góc trên-trái
    panel_h = 160 if telemetry else 40
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def put(text: str, row: int, color=(255, 255, 255)):
        cv2.putText(
            frame, text,
            (8, 20 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )

    # FPS camera
    put(f"CAM  {fps:.1f} fps", 0, (0, 255, 100))

    if telemetry:
        armed_color = (0, 80, 255) if telemetry.armed else (180, 180, 180)
        put(f"MODE {telemetry.flight_mode or '---'}", 1)
        put(f"ARM  {'ARMED' if telemetry.armed else 'DISARMED'}", 2, armed_color)
        put(f"ALT  {telemetry.altitude:.1f} m", 3)
        put(f"SPD  {telemetry.ground_speed:.1f} m/s", 4)
        put(f"BAT  {telemetry.battery_voltage:.1f}V  {telemetry.battery_percent}%", 5,
            (0, 200, 255) if telemetry.battery_percent > 30 else (0, 80, 255))
        put(f"GPS  sats={telemetry.gps_satellites}  fix={telemetry.gps_fix}", 6)

    # Crosshair trung tâm nhỏ (FPV style)
    cx, cy = w // 2, h // 2
    col = (0, 255, 0)
    cv2.line(frame, (cx - 20, cy), (cx - 6, cy), col, 1)
    cv2.line(frame, (cx + 6, cy), (cx + 20, cy), col, 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy - 6), col, 1)
    cv2.line(frame, (cx, cy + 6), (cx, cy + 20), col, 1)
    cv2.circle(frame, (cx, cy), 3, col, 1)

    # Watermark góc dưới phải
    cv2.putText(
        frame, "DRONEEXE FPV",
        (w - 148, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1,
    )

    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="DRONEEXE FPV Camera Demo")
    parser.add_argument(
        "--camera",
        default="https://192.168.4.2:81/stream",
        help="URL stream ESP32-CAM (mặc định: https://192.168.4.2:81/stream)",
    )
    parser.add_argument(
        "--connection",
        default="",
        help="MAVLink connection string (bỏ qua nếu chỉ xem camera)",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Bỏ qua kiểm tra SSL cert (cần khi dùng HTTPS với self-signed cert)",
    )
    args = parser.parse_args()

    # ── Setup camera handler ──
    cam = CameraHandler(
        stream_url=args.camera,
        verify_ssl=not args.no_ssl_verify,
    )

    # ── Setup MAVLink (tùy chọn) ──
    drone: Optional[MAVLinkHandler] = None
    if args.connection:
        drone = MAVLinkHandler(DroneType.ARDUPILOT)
        print(f"🔌 Kết nối MAVLink: {args.connection}")
        connected = await drone.connect(args.connection)
        if connected:
            print("✅ MAVLink connected")
        else:
            print("⚠️  MAVLink không kết nối được — chỉ hiển thị camera")
            drone = None

    # ── Start camera ──
    print(f"📹 Khởi động stream: {args.camera}")
    ok = await cam.start()
    if not ok:
        print("❌ Không khởi động được camera handler")
        return

    print("✅ Stream đang chạy. Nhấn 'q' để thoát, 's' để chụp ảnh.")
    print()

    snapshot_count = 0

    try:
        while True:
            frame = cam.latest_frame
            if frame is not None:
                tel = drone.telemetry if drone else None
                display = build_overlay(frame.copy(), tel, cam.fps)
                cv2.imshow("DRONEEXE FPV", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s") and cam.latest_frame is not None:
                # Chụp ảnh
                fname = f"snapshot_{snapshot_count:03d}.jpg"
                cv2.imwrite(fname, cam.latest_frame)
                snapshot_count += 1
                print(f"📸 Đã lưu: {fname}")

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        await cam.stop()
        if drone:
            await drone.disconnect()
        cv2.destroyAllWindows()
        print("🔌 Đã dừng")


if __name__ == "__main__":
    asyncio.run(main())
