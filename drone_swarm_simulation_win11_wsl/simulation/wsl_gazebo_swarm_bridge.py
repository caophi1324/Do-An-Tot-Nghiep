#!/usr/bin/env python3
"""
WSL2 Gazebo Harmonic <--> ArduPilot Swarm SITL Real-time Bridge
==============================================================
Cầu nối truyền toạ độ trực tiếp từ cụm 5 Drone ArduPilot SITL
sang không gian 3D Gazebo Sim (Harmonic) qua giao thức Gazebo Transport.
- Đồng bộ toạ độ 100% chuẩn xác với giao diện GCS App (Radar 2D & Telemetry).
- Tự động nhận diện IP Host Windows (Default Gateway qua ip route).
- Cập nhật đồng loạt 5 drone qua service set_pose_vector tại 30 FPS siêu mượt.
"""

import math
import os
import sys
import time
import socket
import subprocess
import threading
from typing import Dict, Tuple

try:
    from pymavlink import mavutil
except ImportError:
    print("[ERROR] Cần cài đặt pymavlink trong WSL: pip install pymavlink")
    sys.exit(1)

try:
    from gz.msgs10 import pose_pb2, pose_v_pb2, boolean_pb2
    import gz.transport13 as transport
except ImportError:
    print("[ERROR] Cần cài đặt Gazebo Harmonic Python bindings: sudo apt install python3-gz-transport13 python3-gz-msgs10")
    sys.exit(1)


# Toạ độ bãi đỗ mặt đất chuẩn như trong App GCS (bước 1.6m, dàn hàng trên trục Y)
SWARM_GROUND_OFFSETS = {
    1: (0.0, 0.0),    # Drone 1 (Đỏ)       X=0.0m, Y=0.0m
    2: (0.0, 1.6),    # Drone 2 (Xanh lá)  X=0.0m, Y=1.6m
    3: (0.0, 3.2),    # Drone 3 (Xanh lam) X=0.0m, Y=3.2m
    4: (0.0, 4.8),    # Drone 4 (Vàng)     X=0.0m, Y=4.8m
    5: (0.0, 6.4),    # Drone 5 (Tím)      X=0.0m, Y=6.4m
}


def get_windows_host_ip() -> str:
    """Lấy IP chính xác của Host Windows từ WSL2."""
    # 1. Thử 127.0.0.1 (WSL2 Mirrored Networking)
    for p in (5772, 5770):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        if s.connect_ex(('127.0.0.1', p)) == 0:
            s.close()
            return '127.0.0.1'
        s.close()

    # 2. Lấy default gateway từ 'ip route' (Chuẩn nhất cho WSL2 NAT mode)
    try:
        out = subprocess.check_output("ip route show default", shell=True, text=True)
        parts = out.split()
        if 'via' in parts:
            gw = parts[parts.index('via') + 1]
            return gw
    except Exception:
        pass

    # 3. Fallback đọc /etc/resolv.conf
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if 'nameserver' in line:
                    return line.split()[1]
    except Exception:
        pass

    return '127.0.0.1'


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Chuyển đổi góc Euler (radians) sang Quaternion (qx, qy, qz, qw)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class DroneState:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        # Khởi tạo toạ độ ban đầu đúng vị trí bãi đỗ trong App (không bị trùng nhau ở 0,0)
        ox, oy = SWARM_GROUND_OFFSETS.get(drone_id, (0.0, float(drone_id - 1) * 1.6))
        self.x = ox
        self.y = oy
        self.z = 0.05
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        self.connected = False
        self.last_update = 0.0


class GazeboSwarmBridge:
    def __init__(self, host_ip: str, num_drones: int = 3):
        self.host_ip = host_ip
        self.num_drones = num_drones
        self.states: Dict[int, DroneState] = {i: DroneState(i) for i in range(1, num_drones + 1)}
        self.running = True
        self.gz_node = transport.Node()
        self.service_vec_name = "/world/default/set_pose_vector"

    def mavlink_worker(self, drone_id: int, port: int):
        """Luồng đọc telemetry MAVLink liên tục cho 1 Drone."""
        conn_str = f"tcp:{self.host_ip}:{port}"
        ox, oy = SWARM_GROUND_OFFSETS.get(drone_id, (0.0, float(drone_id - 1) * 1.6))

        while self.running:
            mav = None
            try:
                # autoreconnect=False và retries=0 để tránh pymavlink in rác "[Errno 111] Connection refused sleeping"
                mav = mavutil.mavlink_connection(conn_str, autoreconnect=False, retries=0)
                hb = mav.wait_heartbeat(timeout=3.0)
                if not hb:
                    mav.close()
                    time.sleep(1.0)
                    continue

                self.states[drone_id].connected = True

                # Yêu cầu dòng dữ liệu telemetry 20Hz
                mav.mav.request_data_stream_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1
                )

                while self.running:
                    msg = mav.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE'], blocking=True, timeout=1.0)
                    if msg is None:
                        continue

                    m_type = msg.get_type()
                    now = time.time()
                    st = self.states[drone_id]
                    st.connected = True

                    if m_type == 'LOCAL_POSITION_NED':
                        # Chuyển đổi từ NED sang toạ độ App GCS:
                        # App X = East (msg.y) + ox
                        # App Y = North (msg.x) + oy
                        # App Z = Altitude (-msg.z)
                        st.x = round(float(msg.y) + ox, 3)
                        st.y = round(float(msg.x) + oy, 3)
                        st.z = max(0.05, round(float(-msg.z), 3))
                        st.last_update = now

                    elif m_type == 'ATTITUDE':
                        qx, qy, qz, qw = rpy_to_quaternion(msg.roll, -msg.pitch, -msg.yaw)
                        st.qx, st.qy, st.qz, st.qw = qx, qy, qz, qw
                        st.last_update = now

            except Exception:
                self.states[drone_id].connected = False
                if mav:
                    try:
                        mav.close()
                    except Exception:
                        pass
                time.sleep(1.0)

    def start(self):
        # Khởi chạy 5 thread đọc MAVLink qua cổng SERIAL1 (+2)
        threads = []
        for i in range(1, self.num_drones + 1):
            port = 5760 + i * 10 + 2
            t = threading.Thread(target=self.mavlink_worker, args=(i, port), daemon=True)
            t.start()
            threads.append(t)

        print("\n" + "=" * 65)
        print("🚀 GAZEBO HARMONIC SWARM BRIDGE ĐANG CHẠY (30 FPS)")
        print(f"   Target Service: {self.service_vec_name}")
        print(f"   Windows Host IP: {self.host_ip}")
        print("   Đồng bộ toạ độ: 100% chuẩn theo App GCS Radar")
        print("   Nhấn Ctrl+C để dừng cầu nối.")
        print("=" * 65 + "\n")

        step_counter = 0
        try:
            while self.running:
                pv = pose_v_pb2.Pose_V()
                for i in range(1, self.num_drones + 1):
                    st = self.states[i]
                    p = pv.pose.add()
                    p.name = f"drone{i}"
                    p.position.x = st.x
                    p.position.y = st.y
                    p.position.z = st.z
                    p.orientation.x = st.qx
                    p.orientation.y = st.qy
                    p.orientation.z = st.qz
                    p.orientation.w = st.qw

                try:
                    self.gz_node.request(self.service_vec_name, pv, pose_v_pb2.Pose_V, boolean_pb2.Boolean, 20)
                except Exception:
                    pass

                step_counter += 1
                if step_counter % 30 == 0:
                    info_strs = []
                    for i in range(1, self.num_drones + 1):
                        st = self.states[i]
                        status = "🟢" if st.connected else "🔴"
                        info_strs.append(f"D{i}:{status}({st.x:.1f},{st.y:.1f},{st.z:.1f}m)")
                    print(" | ".join(info_strs))

                time.sleep(1.0 / 30.0)

        except KeyboardInterrupt:
            print("\n[*] Đang tắt Gazebo Swarm Bridge...")
            self.running = False


if __name__ == "__main__":
    host = get_windows_host_ip()
    num_drones = 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        num_drones = int(sys.argv[1])
    bridge = GazeboSwarmBridge(host_ip=host, num_drones=num_drones)
    bridge.start()
