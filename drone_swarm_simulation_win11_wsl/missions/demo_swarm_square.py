import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mavlink_handler import MAVLinkHandler

async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int):
    print(f"[Drone {sysid}] Đang kết nối tới tcp:127.0.0.1:{port}...")
    connected = await handler.connect(f"tcp:127.0.0.1:{port}")
    if not connected:
        print(f"[Drone {sysid}] ❌ Lỗi kết nối!")
        return False
        
    print(f"[Drone {sysid}] ✅ Đã kết nối!")
    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    handler._mavconn.param_set_send("ANGLE_MAX", 6000) # Cho phép nghiêng để tăng tốc
    
    # Khởi đầu với 10 m/s
    handler._mavconn.param_set_send("WPNAV_SPEED", 1000)
    handler._mavconn.param_set_send("AVOID_ENABLE", 0) # Tắt chống va chạm để hạ cánh không bị treo
    
    await asyncio.sleep(1)
    return True

async def main():
    print("========================================")
    print("  TEST: SWARM VUÔNG -> LEO CAO & TĂNG TỐC -> KIM CƯƠNG")
    print("========================================\n")
    
    num_drones = 4
    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    ports = [5772, 5782, 5792, 5802]
    
    setup_tasks = [setup_drone(handlers[i], ports[i], i+1) for i in range(num_drones)]
    results = await asyncio.gather(*setup_tasks)
    
    if not all(results):
        print("Lỗi kết nối. Hãy đảm bảo 4 chiếc drone đang chạy.")
        return
        
    for h in handlers:
        await h.start_telemetry(rate_hz=5)
        
    print("\n⏳ Đang chờ GPS và EKF khởi tạo (10 giây)...")
    await asyncio.sleep(10)
    
    # 1. CẤT CÁNH
    print("\n🚁 Lệnh Swarm: CẤT CÁNH (Bay ở các tầng khác nhau)")
    takeoff_tasks = []
    base_alt = 30.0
    for i, h in enumerate(handlers):
        alt = base_alt + (i * 10) # 30, 40, 50, 60
        takeoff_tasks.append(h.takeoff(altitude=alt))
        
    await asyncio.gather(*takeoff_tasks)
    print("Đang đợi các drone đạt độ cao (8 giây)...")
    await asyncio.sleep(8)
    
    # 2. XẾP HÌNH VUÔNG
    print("\n🔲 Lệnh Swarm: XẾP ĐỘI HÌNH HÌNH VUÔNG (Cùng độ cao 40m)")
    tel1 = handlers[0].telemetry
    center_lat = tel1.latitude
    center_lon = tel1.longitude
    d = 0.00015
    formation_alt = 40.0
    
    square_corners = [
        (center_lat + d, center_lon - d), # Trái trên
        (center_lat + d, center_lon + d), # Phải trên
        (center_lat - d, center_lon - d), # Trái dưới
        (center_lat - d, center_lon + d)  # Phải dưới
    ]
    
    formation_tasks = []
    for i, h in enumerate(handlers):
        lat, lon = square_corners[i]
        formation_tasks.append(h.goto(lat, lon, formation_alt))
        
    await asyncio.gather(*formation_tasks)
    print("Đang tản ra hình vuông... (10 giây)")
    await asyncio.sleep(10)
    
    # 3. DI CHUYỂN, LEO CAO & TĂNG TỐC
    print("\n⬆️ Lệnh Swarm: BẮT ĐẦU DI CHUYỂN PHÍA BẮC VÀ LEO LÊN 70M!")
    new_center_lat = center_lat + 0.0050
    new_center_lon = center_lon
    climb_alt = 70.0
    
    move_tasks = []
    for i, h in enumerate(handlers):
        # Tính điểm đến của từng con để duy trì hình vuông nhưng tịnh tiến lên phía Bắc
        lat, lon = square_corners[i]
        move_tasks.append(h.goto(lat + 0.0050, lon, climb_alt))
        
    await asyncio.gather(*move_tasks)
    
    # Chờ 3 giây rồi bất ngờ tăng tốc độ lên 30m/s!
    print("Đang bay ở 10m/s... (Chờ 3 giây)")
    await asyncio.sleep(3)
    
    print("\n🚀 Lệnh Swarm: BỨT TỐC LÊN 30 M/S VÀO GIỮA KHÔNG TRUNG!!!")
    for h in handlers:
        h._mavconn.param_set_send("WPNAV_SPEED", 3000) # 30 m/s
    
    for _ in range(8):
        speeds = [f"D{i+1}: {h.telemetry.ground_speed:.1f}m/s | Alt: {h.telemetry.altitude:.1f}m" for i, h in enumerate(handlers)]
        print("   " + " || ".join(speeds), end="\r")
        await asyncio.sleep(1)
        
    # 4. CHUYỂN SANG ĐỘI HÌNH KIM CƯƠNG
    print("\n\n💎 Lệnh Swarm: ĐỔI SANG ĐỘI HÌNH KIM CƯƠNG (DIAMOND) ĐỘT NGỘT!")
    diamond_center_lat = new_center_lat + 0.0050 # Tiến tiếp lên Bắc
    diamond_center_lon = new_center_lon
    
    # Kim cương: 1 con đi đầu, 2 con hai bên, 1 con bọc hậu
    d2 = 0.0002
    diamond_corners = [
        (diamond_center_lat + d2, diamond_center_lon),       # D1: Tiên phong (Mũi kim cương)
        (diamond_center_lat, diamond_center_lon + d2),       # D2: Cánh phải
        (diamond_center_lat, diamond_center_lon - d2),       # D3: Cánh trái
        (diamond_center_lat - d2, diamond_center_lon)        # D4: Bọc hậu
    ]
    
    diamond_tasks = []
    for i, h in enumerate(handlers):
        lat, lon = diamond_corners[i]
        diamond_tasks.append(h.goto(lat, lon, climb_alt))
        
    await asyncio.gather(*diamond_tasks)
    
    for _ in range(15):
        speeds = [f"D{i+1}: {h.telemetry.ground_speed:.1f}m/s" for i, h in enumerate(handlers)]
        print("   " + " || ".join(speeds), end="\r")
        await asyncio.sleep(1)
        
    # 5. QUAY VỀ
    print("\n\n🔙 Lệnh Swarm: ĐÁNH LÁI QUAY VỀ (RETURN TO LAUNCH)")
    rtl_tasks = [h.return_to_launch() for h in handlers]
    await asyncio.gather(*rtl_tasks)
    
    print("\nHoàn thành bài test bay nhào lộn nghệ thuật!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
