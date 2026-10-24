import asyncio
import sys
import os
import math

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mavlink_handler import MAVLinkHandler

async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int):
    connected = await handler.connect(f"tcp:127.0.0.1:{port}")
    if not connected: return False
    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    handler._mavconn.param_set_send("WPNAV_SPEED", 2000) # Tốc độ 20m/s
    handler._mavconn.param_set_send("WPNAV_ACCEL", 1000) 
    handler._mavconn.param_set_send("AVOID_ENABLE", 0) # Tắt chống va chạm để hạ cánh không bị treo
    await asyncio.sleep(0.5)
    return True

async def main():
    print("========================================")
    print("  TEST: ĐƯỜNG CONG -> THẲNG 30S -> PHÁO SÁNG")
    print("========================================\n")
    
    num_drones = 4
    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    ports = [5772, 5782, 5792, 5802]
    
    await asyncio.gather(*(setup_drone(handlers[i], ports[i], i+1) for i in range(num_drones)))
    for h in handlers: await h.start_telemetry(rate_hz=5)
    
    print("\n⏳ Đang chờ GPS và EKF khởi tạo (10 giây)...")
    await asyncio.sleep(10)
    
    print("\n🚁 CẤT CÁNH...")
    base_alt = 50.0
    await asyncio.gather(*(h.takeoff(altitude=base_alt) for h in handlers))
    await asyncio.sleep(10)
    
    tel1 = handlers[0].telemetry
    start_lat = tel1.latitude
    start_lon = tel1.longitude
    d = 0.0006  # Tăng khoảng cách đội hình lên 60m để nhìn rõ hơn
    
    # Hàm tính tọa độ đội hình Diamond
    def get_diamond(center_lat, center_lon):
        return [
            (center_lat + d, center_lon),       # D1: Tiên phong
            (center_lat, center_lon + d),       # D2: Cánh phải
            (center_lat, center_lon - d),       # D3: Cánh trái
            (center_lat - d, center_lon)        # D4: Bọc hậu
        ]
        
    print("\n💎 XẾP ĐỘI HÌNH KIM CƯƠNG BẮT ĐẦU VÀO CUA...")
    corners = get_diamond(start_lat, start_lon)
    await asyncio.gather(*(handlers[i].goto(corners[i][0], corners[i][1], base_alt) for i in range(4)))
    await asyncio.sleep(5)
    
    print("\n↪️ BẮT ĐẦU BAY THEO ĐƯỜNG CONG (ARC)")
    # Tính toán 4 điểm trên một cung tròn
    radius = 0.005 # Bán kính đường cong
    for angle_deg in [30, 60, 90, 120]:
        angle_rad = math.radians(angle_deg)
        # Quỹ đạo vòng cung sang phải
        curve_lat = start_lat + (radius * math.sin(angle_rad))
        curve_lon = start_lon + (radius * (1 - math.cos(angle_rad))) 
        
        corners = get_diamond(curve_lat, curve_lon)
        await asyncio.gather(*(handlers[i].goto(corners[i][0], corners[i][1], base_alt) for i in range(4)))
        print(f"   Đang bẻ lái mượt theo đường cong (Góc {angle_deg} độ)...")
        await asyncio.sleep(6) # Liên tục nạp mục tiêu mới để vẽ đường cong
        
    print("\n⬆️ ĐÃ THOÁT CUA! BAY THẲNG ĐỘI HÌNH TRONG 30 GIÂY!")
    # Lấy vị trí tâm hiện tại sau khi đã thoát cua
    current_center_lat = curve_lat
    current_center_lon = curve_lon
    
    # Phóng thẳng đi rất xa (để đảm bảo bay được 30 giây không hết đường)
    straight_lat = current_center_lat - 0.030 
    straight_lon = current_center_lon + 0.010 
    
    corners = get_diamond(straight_lat, straight_lon)
    await asyncio.gather(*(handlers[i].goto(corners[i][0], corners[i][1], base_alt) for i in range(4)))
    
    # Đếm ngược 30 giây bay thẳng
    for sec in range(30):
        speeds = [f"D{i+1}: {h.telemetry.ground_speed:.1f}m/s" for i, h in enumerate(handlers)]
        print(f"[{sec+1}/30s] " + " | ".join(speeds), end="\r")
        await asyncio.sleep(1)
        
    print("\n\n🛑 LỆNH SWARM: ĐỨNG IM (HOVER) GIỮ ĐỘI HÌNH ĐỂ THẢ PHÁO SÁNG")
    # Ra lệnh bay đến chính vị trí GPS hiện tại của từng chiếc để kích hoạt phanh
    for i, h in enumerate(handlers):
        await h.goto(h.telemetry.latitude, h.telemetry.longitude, base_alt)
        
    print("Đang phanh và ổn định đội hình kim cương (5 giây)...")
    await asyncio.sleep(5)
    
    print("\n🎆🎆🎆 KÍCH HOẠT THẢ PHÁO SÁNG ĐỒNG LOẠT 🎆🎆🎆")
    for i, h in enumerate(handlers):
        # Gửi tín hiệu MAVLink kích hoạt Servo số 9 (Dùng để nhả chốt pháo sáng ngoài đời thật)
        from pymavlink import mavutil
        h._mavconn.mav.command_long_send(
            h._mavconn.target_system, h._mavconn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, 9, 1900, 0, 0, 0, 0, 0
        )
    
    # Hiệu ứng Console
    for _ in range(5):
        print("💥 *Bùmmm* Pháo sáng rực rỡ từ 4 Drone!!! 💥")
        await asyncio.sleep(1.5)
        
    print("\n🔙 LỆNH SWARM: HOÀN THÀNH NHIỆM VỤ, TẤT CẢ QUAY VỀ (RTL)")
    await asyncio.gather(*(h.return_to_launch() for h in handlers))

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
