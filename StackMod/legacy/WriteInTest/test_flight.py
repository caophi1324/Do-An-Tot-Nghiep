import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pc_client"))
from drone_mavlink_tracker import DroneMAVLinkTracker

def main():
    print("[FLIGHT TEST] Khởi tạo DroneMAVLinkTracker...")
    tracker = DroneMAVLinkTracker("tcp:127.0.0.1:5773")
    if not tracker.connect():
        print("[FLIGHT TEST] Kết nối thất bại!")
        return
    
    print("[FLIGHT TEST] Chờ đồng bộ Telemetry...")
    time.sleep(1.5)
    print(f"[FLIGHT TEST] Khởi điểm: Mode={tracker.flight_mode}, Alt={tracker.altitude:.2f}m, Armed={tracker.is_armed}")
    
    print("[FLIGHT TEST] Gửi lệnh Cất Cánh (Arm and Takeoff) lên 2.5m...")
    tracker.arm_and_takeoff(target_alt=2.5)
    
    for i in range(12):
        time.sleep(1.0)
        print(f"  T+{i+1}s: Mode={tracker.flight_mode} | Alt={tracker.altitude:.2f}m | Armed={tracker.is_armed}")
        if tracker.altitude >= 1.5:
            print("  ==> Drone đã rời đất và đạt độ cao an toàn!")
            break
            
    cap = cv2.VideoCapture("http://127.0.0.1:8080/stream")
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("gazebo_air_frame.jpg", frame)
        print("[FLIGHT TEST] Đã chụp ảnh từ camera trên không: gazebo_air_frame.jpg")
    
    tracker.close()
    print("[FLIGHT TEST] Hoàn tất.")

if __name__ == "__main__":
    main()
