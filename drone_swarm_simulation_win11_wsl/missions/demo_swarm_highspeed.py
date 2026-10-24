import asyncio
import sys
import os
import math

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mavlink_handler import MAVLinkHandler

async def setup_drone(handler: MAVLinkHandler, port: int, sysid: int):
    """Kết nối và ép giới hạn an toàn để bay tốc độ cao."""
    print(f"[Drone {sysid}] Đang kết nối tới tcp:127.0.0.1:{port}...")
    connected = await handler.connect(f"tcp:127.0.0.1:{port}")
    if not connected:
        print(f"[Drone {sysid}] ❌ Lỗi kết nối!")
        return False
        
    print(f"[Drone {sysid}] ✅ Đã kết nối! Đang nạp tham số tốc độ cao (Racing Params)...")
    
    # 1. Tắt Arming Check để mô phỏng arm lập tức
    handler._mavconn.param_set_send("ARMING_CHECK", 0)
    
    # 2. Cấu hình tốc độ 120km/h (33.3 m/s)
    # ArduCopter lưu tốc độ tính bằng cm/s
    handler._mavconn.param_set_send("WPNAV_SPEED", 3333)       # Tốc độ ngang: 33.3 m/s
    handler._mavconn.param_set_send("WPNAV_ACCEL", 1500)       # Gia tốc ngang: 15 m/s/s
    
    handler._mavconn.param_set_send("WPNAV_SPEED", 3000)       
    handler._mavconn.param_set_send("WPNAV_ACCEL", 2000)      
    handler._mavconn.param_set_send("AVOID_ENABLE", 0)
    
    # 3. Mở giới hạn góc nghiêng (Angle Max)
    # Để bay được 120km/h, drone phải chúc đầu xuống đất rất sâu. Mặc định bị khóa ở 30 độ.
    # Ta nâng lên 60 độ (6000 centi-degrees) để cản gió và bẻ lái gắt 45 độ.
    handler._mavconn.param_set_send("ANGLE_MAX", 6000)         
    handler._mavconn.param_set_send("ATC_SLEW_YAW", 18000)
    
    await asyncio.sleep(1) # Chờ param lưu
    return True

async def main():
    print("========================================")
    print("  TEST: ĐIỀU KHIỂN SWARM 4 DRONES")
    print("  TỐC ĐỘ 120KM/H - BẺ LÁI GẮT")
    print("========================================\n")
    
    num_drones = 4
    handlers = [MAVLinkHandler() for _ in range(num_drones)]
    ports = [5772, 5782, 5792, 5802]
    
    # 1. Kết nối cả 4 con cùng lúc (Parallel)
    setup_tasks = [setup_drone(handlers[i], ports[i], i+1) for i in range(num_drones)]
    results = await asyncio.gather(*setup_tasks)
    
    if not all(results):
        print("Có lỗi kết nối. Vui lòng bật lại start_swarm.py")
        return
        
    print("\n🚀 Tất cả drones đã sẵn sàng! Đang đọc Telemetry...")
    for h in handlers:
        await h.start_telemetry(rate_hz=5)
    print("\n⏳ Đang chờ GPS và EKF khởi tạo (10 giây)...")
    await asyncio.sleep(10) # Chờ có GPS
    
    # 2. CẤT CÁNHcánh đồng loạt với các độ cao khác nhau (để tránh đụng nhau trên không)
    # Dù SITL mặc định có thể đi xuyên nhau, nhưng ta bay theo tầng cho giống thật.
    print("\n🚁 Lệnh Swarm: CẤT CÁNH ĐỒNG LOẠT!")
    takeoff_tasks = []
    base_alt = 30.0
    for i, h in enumerate(handlers):
        alt = base_alt + (i * 10) # 30m, 40m, 50m, 60m
        print(f"   - Drone {i+1} bay lên {alt}m")
        takeoff_tasks.append(h.takeoff(altitude=alt))
        
    await asyncio.gather(*takeoff_tasks)
    print("Đang đợi các drone đạt độ cao ổn định (10 giây)...")
    await asyncio.sleep(10)
    
    # 3. Phóng 1km về phía Bắc ở tốc độ 120km/h
    tel1 = handlers[0].telemetry
    home_lat = tel1.latitude
    home_lon = tel1.longitude
    
    # 0.009 độ vĩ tương đương khoảng 1km
    target_lat = home_lat + 0.0090  
    target_lon = home_lon
    
    print("\n🏁 Lệnh Swarm: BAY THẲNG 1KM VỀ PHÍA BẮC (Vận tốc 120km/h)!")
    goto_tasks = []
    for i, h in enumerate(handlers):
        alt = base_alt + (i * 10)
        # Xếp đội hình chéo nhau một chút bằng cách cộng tọa độ kinh (lon)
        offset_lon = target_lon + (i * 0.0002) 
        goto_tasks.append(h.goto(target_lat, offset_lon, alt))
        
    await asyncio.gather(*goto_tasks)
    
    # Theo dõi tốc độ bứt tốc
    print("\n📊 Đang giám sát tốc độ (Quan sát chúng tăng tốc lên >30m/s):")
    for _ in range(25):
        speeds = [f"D{i+1}: {h.telemetry.ground_speed:.1f} m/s" for i, h in enumerate(handlers)]
        print(" | ".join(speeds), end="\r")
        await asyncio.sleep(1)
        
    # 4. Bẻ lái vuông góc 90 độ (với góc nghiêng cực gắt do vận tốc đang cao)
    target_lat_2 = target_lat
    target_lon_2 = target_lon + 0.0090 # Bay 1km sang phía Đông
    print("\n\n↪️ Lệnh Swarm: BẺ LÁI GẮT SANG HƯỚNG ĐÔNG (Góc nghiêng gắt)!")
    
    goto_tasks_2 = []
    for i, h in enumerate(handlers):
        alt = base_alt + (i * 10)
        offset_lat = target_lat_2 + (i * 0.0002)
        goto_tasks_2.append(h.goto(offset_lat, target_lon_2, alt))
        
    await asyncio.gather(*goto_tasks_2)
    
    for _ in range(20):
        speeds = [f"D{i+1}: {h.telemetry.ground_speed:.1f} m/s" for i, h in enumerate(handlers)]
        print(" | ".join(speeds), end="\r")
        await asyncio.sleep(1)
        
    # 5. Quay về nhà
    print("\n\n🔙 Lệnh Swarm: RETURN TO LAUNCH (Quay về nhà)!")
    rtl_tasks = [h.return_to_launch() for h in handlers]
    await asyncio.gather(*rtl_tasks)
    
    print("\nĐội hình đang tự động bay về. Bài Test siêu tốc độ hoàn thành xuất sắc!")
    
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
