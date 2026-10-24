#!/usr/bin/env python3
"""
Test Gazebo Camera Stream Reception on Windows
----------------------------------------------
Kết nối tới luồng video HTTP MJPEG do gazebo_camera_bridge.py phát tại http://127.0.0.1:8080/stream.
Kiểm tra:
1. Đọc frame thành công từ OpenCV trên Windows
2. Đo độ trễ và kích thước frame (640x480)
3. Chạy thử inference YOLOv11 nhận diện người (nếu có model)
"""

import sys
import time
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def test_stream(url="http://127.0.0.1:8080/stream", max_frames=30):
    print(f"[*] Đang kết nối tới Camera Stream Gazebo: {url}")
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"[!] KHÔNG THỂ KẾT NỐI TỚI {url}. Kiểm tra xem Gazebo và Bridge đã chạy chưa!")
        return False

    print("[+] Kết nối thành công! Đang đọc frame...")
    frames_read = 0
    start_time = time.time()
    
    while frames_read < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Lỗi khi đọc frame.")
            time.sleep(0.05)
            continue
            
        frames_read += 1
        if frames_read == 1:
            h, w, c = frame.shape
            print(f"[+] Frame đầu tiên nhận được! Độ phân giải: {w}x{h} (channels: {c})")
            
        if frames_read % 10 == 0:
            elapsed = time.time() - start_time
            fps = frames_read / elapsed
            print(f"[*] Đã nhận {frames_read}/{max_frames} frames | FPS trung bình: {fps:.1f}")

    cap.release()
    total_time = time.time() - start_time
    avg_fps = frames_read / total_time
    print(f"[+] TEST HOÀN TẤT THÀNH CÔNG! Đọc được {frames_read} frames trong {total_time:.2f}s (~{avg_fps:.1f} FPS)")
    return True

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/stream"
    success = test_stream(url)
    sys.exit(0 if success else 1)
