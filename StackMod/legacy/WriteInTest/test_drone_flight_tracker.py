#!/usr/bin/env python3
"""
UNIT TEST CHO MODULE DRONE MAVLINK TRACKER
Kiểm tra tính đúng đắn toán học của các luật điều khiển bay:
- Follow-Me
- Orbit quanh người
- Bù độ cao
- Lọc thông thấp EMA
- Vùng trễ Deadband
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pc_client"))

from drone_mavlink_tracker import DroneMAVLinkTracker


class MockMaster:
    def __init__(self):
        self.target_system = 1
        self.target_component = 1
        self.last_sent_msg = None
        self.mav = self

    def set_position_target_local_ned_send(self, time_boot_ms, sys_id, comp_id, frame, mask,
                                           x, y, z, vx, vy, vz, afx, afy, afz, yaw, yaw_rate):
        self.last_sent_msg = {
            "frame": frame,
            "mask": mask,
            "vx": vx, "vy": vy, "vz": vz,
            "yaw_rate": yaw_rate
        }


def test_drone_tracker():
    print("[TEST] Bắt đầu kiểm thử DroneMAVLinkTracker...")
    tracker = DroneMAVLinkTracker()
    mock_master = MockMaster()
    tracker.master = mock_master
    tracker.is_connected = True
    tracker.is_armed = True
    tracker.altitude = 2.5  # Đang ở độ cao chuẩn 2.5m

    img_shape = (480, 640)

    # Test 1: Mục tiêu ở chính giữa màn hình, khoảng cách chuẩn (bh = 150)
    # cx = 320, cy = 240, bh = 150
    bbox_center = [320 - 40, 240 - 75, 320 + 40, 240 + 75]
    for _ in range(10):  # Chạy vài bước để EMA hội tụ
        tracker.update_drone_tracking(bbox_center, img_shape, mode="FOLLOW")
    
    msg = mock_master.last_sent_msg
    print(f"Test 1 (Centered Follow): vx={msg['vx']:.3f}, vy={msg['vy']:.3f}, vz={msg['vz']:.3f}, yaw_rate={msg['yaw_rate']:.3f}")
    assert abs(msg['yaw_rate']) < 0.05, "Lỗi: Mục tiêu ở giữa thì yaw_rate phải xấp xỉ 0!"
    assert abs(msg['vx']) < 0.05, "Lỗi: Khoảng cách chuẩn thì vx phải xấp xỉ 0!"
    assert abs(msg['vy']) < 0.05, "Lỗi: Follow mode không có vận tốc ngang vy!"

    # Test 2: Mục tiêu lệch sang PHẢI (cx = 500 > 320)
    bbox_right = [500 - 40, 240 - 75, 500 + 40, 240 + 75]
    for _ in range(10):
        tracker.update_drone_tracking(bbox_right, img_shape, mode="FOLLOW")
    msg = mock_master.last_sent_msg
    print(f"Test 2 (Target Right): yaw_rate={msg['yaw_rate']:.3f} rad/s")
    assert msg['yaw_rate'] > 0.1, "Lỗi: Mục tiêu bên phải thì drone phải quay sang phải (yaw_rate > 0)!"

    # Test 3: Mục tiêu lệch sang TRÁI (cx = 140 < 320)
    bbox_left = [140 - 40, 240 - 75, 140 + 40, 240 + 75]
    for _ in range(10):
        tracker.update_drone_tracking(bbox_left, img_shape, mode="FOLLOW")
    msg = mock_master.last_sent_msg
    print(f"Test 3 (Target Left): yaw_rate={msg['yaw_rate']:.3f} rad/s")
    assert msg['yaw_rate'] < -0.1, "Lỗi: Mục tiêu bên trái thì drone phải quay sang trái (yaw_rate < 0)!"

    # Test 4: Mục tiêu ở XA (bh = 80 < 150) -> Tiến tới (vx > 0)
    bbox_far = [320 - 20, 240 - 40, 320 + 20, 240 + 40]
    for _ in range(10):
        tracker.update_drone_tracking(bbox_far, img_shape, mode="FOLLOW")
    msg = mock_master.last_sent_msg
    print(f"Test 4 (Target Far): vx={msg['vx']:.3f} m/s")
    assert msg['vx'] > 0.2, "Lỗi: Mục tiêu ở xa thì drone phải tiến tới (vx > 0)!"

    # Test 5: Mục tiêu QUÁ GẦN (bh = 240 > 150) -> Lùi lại (vx < 0)
    bbox_close = [320 - 60, 240 - 120, 320 + 60, 240 + 120]
    for _ in range(10):
        tracker.update_drone_tracking(bbox_close, img_shape, mode="FOLLOW")
    msg = mock_master.last_sent_msg
    print(f"Test 5 (Target Close): vx={msg['vx']:.3f} m/s")
    assert msg['vx'] < -0.2, "Lỗi: Mục tiêu quá gần thì drone phải lùi lại (vx < 0)!"

    # Test 6: Chế độ ORBIT -> Có vận tốc ngang vy = orbit_speed (~0.8 m/s)
    for _ in range(10):
        tracker.update_drone_tracking(bbox_center, img_shape, mode="ORBIT")
    msg = mock_master.last_sent_msg
    print(f"Test 6 (Orbit Mode): vy={msg['vy']:.3f} m/s")
    assert msg['vy'] > 0.5, "Lỗi: Chế độ Orbit phải có vận tốc ngang vy để lượn vòng cung!"

    # Test 7: Bù cao độ (Altitude Control): Drone đang thấp ở 1.8m (target 2.5m) -> vz âm để bay LÊN
    tracker.altitude = 1.8
    for _ in range(10):
        tracker.update_drone_tracking(bbox_center, img_shape, mode="FOLLOW")
    msg = mock_master.last_sent_msg
    print(f"Test 7 (Altitude Climb): vz={msg['vz']:.3f} m/s (NED: vz < 0 là bay lên)")
    assert msg['vz'] < -0.2, "Lỗi: Khi thấp hơn độ cao mục tiêu thì vz phải âm để bay lên!"

    # Test 8: Chế độ HOVER hoặc mất dấu (bbox is None) -> Vận tốc trả về 0
    for _ in range(10):
        tracker.update_drone_tracking(None, img_shape, mode="HOVER")
    msg = mock_master.last_sent_msg
    print(f"Test 8 (Hover/Lost): vx={msg['vx']:.3f}, vy={msg['vy']:.3f}, vz={msg['vz']:.3f}")
    assert abs(msg['vx']) < 0.05 and abs(msg['vy']) < 0.05 and abs(msg['vz']) < 0.05, "Lỗi: Hover phải dừng lơ lửng!"

    print("\n✅ TOÀN BỘ 8/8 BÀI TEST TOÁN HỌC & ĐIỀU KHIỂN BAY ĐÃ PASS XUẤT SẮC!")


if __name__ == "__main__":
    test_drone_tracker()
