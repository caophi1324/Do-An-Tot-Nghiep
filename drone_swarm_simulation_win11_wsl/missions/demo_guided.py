"""
DRONEEXE — Task 6 Demo: MAVLink Guided Mode Control

Script demo điều khiển drone qua Guided mode.
Chạy với SITL: python scripts/demo_guided.py --sitl
Chạy với drone thật: python scripts/demo_guided.py --connection serial:COM3:57600

Yêu cầu: ArduPilot SITL đang chạy hoặc drone thật đã kết nối.
"""

import asyncio
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mavlink_handler import MAVLinkHandler
from core.base_handler import FlightMode, DroneType


async def demo_basic_control(handler: MAVLinkHandler):
    """Demo 1: Cất cánh, hover, hạ cánh."""
    print("\n" + "=" * 60)
    print("  DEMO 1: Basic Control — Takeoff, Hover, Land")
    print("=" * 60)

    # Start telemetry
    await handler.start_telemetry(rate_hz=5)
    await asyncio.sleep(2)  # Chờ telemetry ổn định

    tel = handler.telemetry
    print(f"\n📡 Telemetry:")
    print(f"   GPS: ({tel.latitude:.6f}, {tel.longitude:.6f})")
    print(f"   Alt: {tel.altitude:.1f}m | Battery: {tel.battery_percent}%")
    print(f"   Mode: {tel.flight_mode} | Armed: {tel.armed}")

    # Takeoff
    print("\n🚁 Taking off to 5m...")
    success = await handler.takeoff(altitude=5.0)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")

    if success:
        # Wait for altitude
        print("   Climbing...")
        for i in range(15):
            await asyncio.sleep(1)
            tel = handler.telemetry
            print(f"   Alt: {tel.altitude:.1f}m", end="\r")
            if tel.altitude > 4.5:
                break
        print(f"\n   Reached {tel.altitude:.1f}m")

        # Hover for 5 seconds
        print("\n⏸️  Hovering for 5 seconds...")
        await asyncio.sleep(5)

        # Land
        print("\n🛬 Landing...")
        await handler.land()
        await asyncio.sleep(10)
        print(f"   Final alt: {handler.telemetry.altitude:.1f}m")

    await handler.stop_telemetry()


async def demo_guided_goto(handler: MAVLinkHandler):
    """Demo 2: Bay tới tọa độ GPS trong Guided mode."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Guided Mode — Goto GPS Coordinates")
    print("=" * 60)

    await handler.start_telemetry(rate_hz=5)
    await asyncio.sleep(2)

    # Lấy vị trí hiện tại làm home
    tel = handler.telemetry
    home_lat = tel.latitude
    home_lon = tel.longitude
    print(f"\n🏠 Home position: ({home_lat:.6f}, {home_lon:.6f})")

    # Takeoff
    print("\n🚁 Taking off to 30m...")
    await handler.takeoff(altitude=30.0)
    await asyncio.sleep(15)

    # Di chuyển khoảng 200m về phía Bắc (tăng latitude ~0.0018)
    target_lat = home_lat + 0.0018
    target_lon = home_lon
    target_alt = 30.0

    print(f"\n📍 Going to ({target_lat:.6f}, {target_lon:.6f}) at {target_alt}m...")
    await handler.goto(target_lat, target_lon, target_alt)

    # Monitor progress (đợi tối đa 60 giây vì bay xa hơn)
    for i in range(60):
        await asyncio.sleep(1)
        tel = handler.telemetry
        distance = _haversine(tel.latitude, tel.longitude, target_lat, target_lon)
        print(f"   Distance: {distance:.1f}m | Alt: {tel.altitude:.1f}m | Speed: {tel.ground_speed:.1f}m/s", end="\r")
        if distance < 2.0:
            print(f"\n   ✅ Arrived at target!")
            break

    # Return to launch
    print("\n🔙 Returning to launch...")
    await handler.return_to_launch()
    await asyncio.sleep(15)

    print(f"   Final position: ({handler.telemetry.latitude:.6f}, {handler.telemetry.longitude:.6f})")
    await handler.stop_telemetry()


async def demo_velocity_control(handler: MAVLinkHandler):
    """Demo 3: Điều khiển bằng velocity (cho companion computer)."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Velocity Control — Move by speed commands")
    print("=" * 60)

    await handler.start_telemetry(rate_hz=10)
    await asyncio.sleep(2)

    print("\n🚁 Taking off to 5m...")
    await handler.takeoff(altitude=5.0)
    await asyncio.sleep(8)

    # Move forward at 2 m/s for 5 seconds
    print("\n➡️  Moving forward at 2 m/s for 5 seconds...")
    for _ in range(50):  # 50 * 0.1s = 5s
        await handler.set_velocity(vx=2.0, vy=0, vz=0)
        await asyncio.sleep(0.1)

    # Stop
    print("⏹️  Stopping...")
    await handler.set_velocity(vx=0, vy=0, vz=0)
    await asyncio.sleep(2)

    # Move right at 1.5 m/s for 3 seconds
    print("➡️  Moving right at 1.5 m/s for 3 seconds...")
    for _ in range(30):
        await handler.set_velocity(vx=0, vy=1.5, vz=0)
        await asyncio.sleep(0.1)

    # Stop and hover
    await handler.set_velocity(vx=0, vy=0, vz=0)
    print("⏸️  Hovering...")
    await asyncio.sleep(3)

    # Yaw rotation
    print("🔄 Rotating 90° clockwise...")
    await handler.move_relative(0, 0, 0, yaw=90)
    await asyncio.sleep(5)

    # Land
    print("\n🛬 Landing...")
    await handler.land()
    await asyncio.sleep(10)

    await handler.stop_telemetry()


async def demo_mode_switching(handler: MAVLinkHandler):
    """Demo 4: Chuyển đổi flight mode."""
    print("\n" + "=" * 60)
    print("  DEMO 4: Flight Mode Switching")
    print("=" * 60)

    await handler.start_telemetry(rate_hz=5)
    await asyncio.sleep(2)

    modes_to_test = [
        FlightMode.STABILIZE,
        FlightMode.ALT_HOLD,
        FlightMode.LOITER,
        FlightMode.GUIDED,
    ]

    for mode in modes_to_test:
        print(f"\n🔄 Switching to {mode.value}...")
        success = await handler.set_mode(mode)
        current = await handler.get_mode()
        status = "✅" if success else "❌"
        print(f"   {status} Current mode: {current.value}")
        await asyncio.sleep(1)

    await handler.stop_telemetry()


async def main():
    parser = argparse.ArgumentParser(description="DRONEEXE Guided Mode Demo")
    parser.add_argument("--sitl", action="store_true", help="Connect to SITL (udp:127.0.0.1:14550)")
    parser.add_argument("--connection", type=str, default="", help="Connection string (e.g., serial:COM3:57600)")
    parser.add_argument("--demo", type=int, default=0, help="Demo number (1-4, 0=all)")
    args = parser.parse_args()

    # Determine connection string
    if args.sitl:
        conn_str = "tcp:127.0.0.1:5762"
    elif args.connection:
        conn_str = args.connection
    else:
        print("Usage:")
        print("  python scripts/demo_guided.py --sitl")
        print("  python scripts/demo_guided.py --connection serial:COM3:57600")
        return

    # Connect
    handler = MAVLinkHandler(DroneType.ARDUPILOT)
    print(f"🔌 Connecting to {conn_str}...")
    connected = await handler.connect(conn_str)

    if not connected:
        print("❌ Connection failed!")
        return

    print("✅ Connected!")

    try:
        demos = {
            1: ("Basic Control", demo_basic_control),
            2: ("Guided Goto", demo_guided_goto),
            3: ("Velocity Control", demo_velocity_control),
            4: ("Mode Switching", demo_mode_switching),
        }

        if args.demo == 0:
            # Run mode switching first (safe, no flight)
            await demo_mode_switching(handler)
        elif args.demo in demos:
            name, func = demos[args.demo]
            await func(handler)
        else:
            print(f"Unknown demo: {args.demo}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Landing...")
        await handler.land()
    finally:
        await handler.disconnect()
        print("\n🔌 Disconnected")


def _haversine(lat1, lon1, lat2, lon2):
    """Distance between two GPS coords in meters."""
    import math
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


if __name__ == "__main__":
    asyncio.run(main())
