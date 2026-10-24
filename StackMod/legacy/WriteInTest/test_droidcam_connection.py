"""
Kiểm tra kết nối trực tiếp DroidCam qua WiFi
============================================
Chạy: python test_droidcam_connection.py
"""

import cv2
import time
import sys
import io

import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Nhận IP từ dòng lệnh (mặc định: 192.168.1.68)
target_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.68"
if not target_ip.startswith("http"):
    DROIDCAM_URL = f"http://{target_ip}:4747/video"
else:
    DROIDCAM_URL = target_ip

print("=" * 65)
print(f"📡 ĐANG KIỂM TRA KẾT NỐI DROIDCAM TẠI: {DROIDCAM_URL}")
print("=" * 65)

t_start = time.time()
cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    print("❌ KHÔNG THỂ KẾT NỐI ĐẾN DROIDCAM!")
    print("\n💡 Các bước kiểm tra nhanh:")
    print("1. Hãy chắc chắn app DroidCam trên điện thoại đang MỞ và hiển thị số IP/Port 4747.")
    print("2. Máy tính và điện thoại phải cùng bắt MỘT mạng WiFi (không bật 4G/VPN).")
    print(f"3. Kiểm tra địa chỉ IP trên điện thoại có đúng là {target_ip} không.")
    print(f"   URL dự kiến: {DROIDCAM_URL}")
    sys.exit(1)

print("✅ Đã kết nối thành công tới DroidCam stream qua OpenCV!")
print("📸 Đang đọc thử 10 frames và đo độ trễ/FPS...")

success_frames = 0
resolutions = set()
frame_times = []
last_frame = None

for i in range(10):
    t0 = time.time()
    ret, frame = cap.read()
    dt = time.time() - t0
    if ret and frame is not None:
        h, w = frame.shape[:2]
        resolutions.add((w, h))
        success_frames += 1
        frame_times.append(dt)
        last_frame = frame
        print(f"  • Frame #{i+1:02d}: Độ phân giải {w}x{h} px | Thời gian đọc: {dt*1000:.1f}ms - OK")
    else:
        print(f"  • Frame #{i+1:02d}: Không nhận được frame")
    time.sleep(0.02)

cap.release()

if success_frames > 0:
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    res_str = ", ".join([f"{w}x{h}" for w, h in resolutions])
    
    # Lưu frame kiểm tra để xác nhận hình ảnh
    save_path = "droidcam_test_frame.jpg"
    if last_frame is not None:
        cv2.imwrite(save_path, last_frame)
        print(f"\n🖼️ Đã lưu ảnh chụp thử nghiệm từ DroidCam vào: {os.path.abspath(save_path)}")

    print("\n" + "=" * 65)
    print("🎉 DROIDCAM HOẠT ĐỘNG HOÀN HẢO!")
    print(f"  • Tổng số frame nhận được : {success_frames}/10")
    print(f"  • Độ phân giải            : {res_str}")
    print(f"  • Tốc độ đọc ước tính     : ~{avg_fps:.1f} FPS")
    print("=" * 65)
    print("\n👉 Để dùng luồng DroidCam này cho AI Tracking PC Client:")
    print(f"   Cách 1: Khởi động app: python pc_client/app.py")
    print(f"           Trong giao diện UI, nhập URL: {DROIDCAM_URL}")
    print(f"   Cách 2: Hoặc đặt 'custom_url: \"{DROIDCAM_URL}\"' trong pc_client/config.yaml")
else:
    print("\n⚠️ Kết nối được tới cổng nhưng không giải mã được frame ảnh nào từ DroidCam.")

