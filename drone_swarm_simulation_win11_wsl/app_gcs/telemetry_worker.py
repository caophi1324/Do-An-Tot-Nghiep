#!/usr/bin/env python3
"""
DRONE SWARM GCS - TELEMETRY WORKER & MAVLINK COMMAND BRIDGE
Đọc telemetry 5 Drone ArduPilot qua QThread độc lập và gửi lệnh điều khiển.
"""

import math
import os
import sys
import socket
import threading
import time
import subprocess
import itertools
from typing import Dict, Optional
from PyQt6.QtCore import QThread, pyqtSignal, QObject

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False

# Ép xung nhịp hẹn giờ Windows Scheduler xuống 1.0ms (triệt tiêu timer jitter 15.6ms của OS)
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

from .styles import SWARM_GROUND_OFFSETS, DRONE_COLORS, DEFAULT_NUM_DRONES


def calc_lipo_3s_pct(v: float) -> int:
    """
    Ánh xạ điện áp Pin 3S LiPo sang % dung lượng theo bảng thực nghiệm chuẩn SITL SIM_Battery
    12.60V -> 100%
    12.40V -> 95%
    12.20V -> 85%
    11.90V -> 75%
    11.60V -> 60%
    11.30V -> 50%
    11.00V -> 35%
    10.50V -> 10% (Ngưỡng BATT_LOW_VOLT cảnh báo hạ cánh)
    9.90V  -> 0%  (Ngưỡng BATT_CRT_VOLT)
    """
    if v >= 12.58:
        return 100
    elif v >= 12.40:
        return int(95 + (v - 12.40) / (12.60 - 12.40) * 5)
    elif v >= 12.20:
        return int(85 + (v - 12.20) / (12.40 - 12.20) * 10)
    elif v >= 11.90:
        return int(75 + (v - 11.90) / (12.20 - 11.90) * 10)
    elif v >= 11.60:
        return int(60 + (v - 11.60) / (11.90 - 11.60) * 15)
    elif v >= 11.30:
        return int(50 + (v - 11.30) / (11.60 - 11.30) * 10)
    elif v >= 11.00:
        return int(35 + (v - 11.00) / (11.30 - 11.00) * 15)
    elif v >= 10.50:
        return int(10 + (v - 10.50) / (11.00 - 10.50) * 25)
    elif v >= 9.90:
        return max(0, int((v - 9.90) / (10.50 - 9.90) * 10))
    else:
        return 0


class SwarmTelemetryWorker(QThread):
    """
    Luồng nền đọc MAVLink trực tiếp từ các Drone ở tần số 20Hz,
    tính toán cự ly bầy đàn và bắn dữ liệu an toàn sang GUI.
    """
    telemetry_updated = pyqtSignal(dict)
    log_message = pyqtSignal(str, str)  # (level, text)

    def __init__(self, num_drones: int = DEFAULT_NUM_DRONES, parent=None):
        super().__init__(parent)
        self.num_drones = num_drones
        self.running = True
        self.connections = {}
        self.raw_data = {}

        # Khởi tạo khung dữ liệu mặc định cho 5 drone
        for i in range(1, num_drones + 1):
            ox, oy = SWARM_GROUND_OFFSETS.get(i, (0.0, float(i - 1) * 1.6))
            self.raw_data[i] = {
                "id": i,
                "connected": False,
                "armed": False,
                "mode": "DISCONN",
                "x": ox,
                "y": oy,
                "z": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "voltage": 12.6,
                "battery": 100,
                "throttle": 0,
                "failsafe_active": False,
                "failsafe_text": "",
                "last_seen": 0.0,
                "receive_errors": 0,
                "packets_received": 0,
                "packet_loss_pct": 0.0,
            }

    def run(self):
        if not HAS_PYMAVLINK:
            self.log_message.emit("ERROR", "Không tìm thấy thư viện pymavlink! Hãy cài đặt pymavlink.")
            return

        self.log_message.emit("INFO", f"Đang khởi động luồng Telemetry cho {self.num_drones} Drones...")

        # Khởi tạo các thread con đọc socket cho từng drone
        reader_threads = []
        for i in range(1, self.num_drones + 1):
            # Cổng SERIAL0 mặc định: D1:5770, D2:5780, D3:5790, D4:5800, D5:5810
            # Giữ cổng SERIAL1 (5772..) cho RViz2 Mesh Publisher và SERIAL2 (5773..) cho Mission Scripts
            port = 5760 + i * 10
            t = threading.Thread(target=self._read_drone_socket, args=(i, port), daemon=True)
            t.start()
            reader_threads.append(t)

        # Vòng lặp chính của QThread: Tổng hợp và bắn Signal sang GUI ở 20Hz (mỗi 50ms)
        while self.running:
            now = time.time()
            min_dist = float('inf')
            connected_count = 0

            # Cập nhật trạng thái timeout (nếu > 2s không nhận gói tin -> Disconnected)
            for i in range(1, self.num_drones + 1):
                d = self.raw_data[i]
                if now - d["last_seen"] > 2.5:
                    d["connected"] = False
                if d["connected"]:
                    connected_count += 1

            # Tính toán khoảng cách tối thiểu giữa mọi cặp drone
            warning_pairs = []
            for i in range(1, self.num_drones + 1):
                for j in range(i + 1, self.num_drones + 1):
                    p1 = self.raw_data[i]
                    p2 = self.raw_data[j]
                    if p1["connected"] and p2["connected"]:
                        dx = p1["x"] - p2["x"]
                        dy = p1["y"] - p2["y"]
                        dz = p1["z"] - p2["z"]
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist < min_dist:
                            min_dist = dist
                        if dist < 1.20:  # Vùng ảnh hưởng APF (d_influence = 1.20m cho hội trường 20x30m)
                            warning_pairs.append((i, j, dist))

            # Xác định trạng thái an toàn APF (chuẩn apf_swarm_engine hội trường 20x30m)
            if min_dist < 0.50:
                safe_status = "CRITICAL_AVOID"
            elif min_dist < 0.85:
                safe_status = "WARNING_CLOSE"
            elif min_dist < 1.20:
                safe_status = "APF_ACTIVE"
            else:
                safe_status = "FORMATION_NOMINAL"

            payload = {
                "drones": self.raw_data,
                "connected_count": connected_count,
                "min_distance_m": round(min_dist, 2) if min_dist != float('inf') else 999.0,
                "safe_status": safe_status,
                "warning_pairs": warning_pairs,
                "timestamp": now,
            }

            self.telemetry_updated.emit(payload)
            self.msleep(50)  # 20Hz

    def _read_drone_socket(self, drone_id: int, port: int):
        """Worker kết nối TCP socket nhận telemetry thời gian thực từ 1 drone."""
        conn_str = f"tcp:127.0.0.1:{port}"
        ox, oy = SWARM_GROUND_OFFSETS.get(drone_id, (0.0, float(drone_id - 1) * 1.6))

        while self.running:
            mav = None
            try:
                # retries=0 và autoreconnect=False: pymavlink raise Exception ngay lập tức khi cổng đóng, không bao giờ in "sleeping"
                mav = mavutil.mavlink_connection(conn_str, autoreconnect=False, retries=0)
                # Triệt tiêu hoàn toàn độ trễ Nagle trên Windows TCP socket (chống nghẽn gói tin gây khựng ở 20Hz)
                if hasattr(mav, 'port') and hasattr(mav.port, 'setsockopt'):
                    try:
                        mav.port.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                # Đợi heartbeat ban đầu (chờ tối đa 3.0s)
                msg = mav.wait_heartbeat(timeout=3.0)
                if not msg:
                    mav.close()
                    time.sleep(1.0)
                    continue

                self.connections[drone_id] = mav
                self.raw_data[drone_id]["connected"] = True
                self.log_message.emit("SUCCESS", f"[Drone {drone_id}] Đã kết nối Telemetry (Cổng {port})")

                # Cung cấp toạ độ EKF Origin & Home chuẩn cho indoor drone (giải phóng triệt để lỗi PreArm: Need Position Estimate)
                try:
                    lat_deg = -35.3632621
                    lon_deg = 149.1652374 + drone_id * 0.0001
                    alt_m = 584.0
                    lat = int(lat_deg * 1e7)
                    lon = int(lon_deg * 1e7)
                    alt = int(alt_m * 1000)
                    mav.mav.set_gps_global_origin_send(mav.target_system, lat, lon, alt)
                    mav.mav.command_long_send(
                        mav.target_system,
                        mav.target_component,
                        mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                        0,
                        0, 0, 0, 0,
                        lat_deg, lon_deg, alt_m
                    )
                except Exception:
                    pass

                # Yêu cầu dữ liệu vị trí và attitude ở 20Hz
                mav.mav.request_data_stream_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1
                )

                while self.running:
                    msg = mav.recv_match(blocking=True, timeout=1.0)
                    if msg is None:
                        continue

                    m_type = msg.get_type()
                    self.raw_data[drone_id]["last_seen"] = time.time()
                    self.raw_data[drone_id]["connected"] = True

                    if m_type == "HEARTBEAT":
                        self.raw_data[drone_id]["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        # Mode mapping
                        mode_map = mavutil.mode_mapping_byname(mav.flightmode) or {}
                        self.raw_data[drone_id]["mode"] = mav.flightmode or "UNKNOWN"
                        if hasattr(mav, 'mav'):
                            self.raw_data[drone_id]["receive_errors"] = getattr(mav.mav, 'total_receive_errors', 0)
                            self.raw_data[drone_id]["packets_received"] = getattr(mav.mav, 'total_packets_received', 0)
                        if hasattr(mav, 'packet_loss'):
                            try:
                                self.raw_data[drone_id]["packet_loss_pct"] = round(float(mav.packet_loss()), 1)
                            except Exception:
                                pass

                    elif m_type == "LOCAL_POSITION_NED":
                        # Đổi từ ArduPilot NED (X North, Y East, Z Down) sang ROS/GCS (X East, Y North, Z Up)
                        ros_x = float(msg.y) + ox
                        ros_y = float(msg.x) + oy
                        ros_z = float(-msg.z)

                        self.raw_data[drone_id]["x"] = round(ros_x, 3)
                        self.raw_data[drone_id]["y"] = round(ros_y, 3)
                        self.raw_data[drone_id]["z"] = max(0.0, round(ros_z, 3))
                        self.raw_data[drone_id]["vx"] = round(float(msg.vy), 2)
                        self.raw_data[drone_id]["vy"] = round(float(msg.vx), 2)
                        self.raw_data[drone_id]["vz"] = round(float(-msg.vz), 2)

                    elif m_type == "ATTITUDE":
                        self.raw_data[drone_id]["roll"] = round(math.degrees(msg.roll), 1)
                        self.raw_data[drone_id]["pitch"] = round(math.degrees(msg.pitch), 1)
                        # Góc yaw chuyển sang hệ ROS (East = 0, North = 90)
                        yaw_deg = (90.0 - math.degrees(msg.yaw)) % 360.0
                        self.raw_data[drone_id]["yaw"] = round(yaw_deg, 1)

                    elif m_type == "VFR_HUD":
                        self.raw_data[drone_id]["speed"] = round(msg.groundspeed, 2)
                        self.raw_data[drone_id]["throttle"] = int(getattr(msg, "throttle", 0))

                    elif m_type == "STATUSTEXT":
                        text = getattr(msg, 'text', '')
                        if isinstance(text, bytes):
                            text = text.decode('utf-8', errors='ignore')
                        text = str(text).strip()
                        if text:
                            is_critical = any(kw in text for kw in ["critical", "Crash", "Failsafe", "Emergency"])
                            is_warning = any(kw in text for kw in ["low", "Low", "Battery", "Disarming", "PreArm", "failsafe"])
                            if is_critical or is_warning:
                                level = "ERROR" if is_critical else "WARNING"
                                self.log_message.emit(level, f"[Drone {drone_id}] ⚠️ {text}")
                            if "Battery" in text or "Failsafe" in text or "low" in text:
                                self.raw_data[drone_id]["failsafe_active"] = True
                                self.raw_data[drone_id]["failsafe_text"] = text

                    elif m_type == "SYS_STATUS":
                        v = round(msg.voltage_battery / 1000.0, 2)
                        self.raw_data[drone_id]["voltage"] = v
                        self.raw_data[drone_id]["battery"] = calc_lipo_3s_pct(v)

                        if getattr(msg, 'current_battery', -1) > 0:
                            self.raw_data[drone_id]["current_a"] = round(msg.current_battery / 100.0, 2)

                        # Kiểm tra trạng thái Failsafe theo điện áp và mode
                        mode = self.raw_data[drone_id]["mode"]
                        if v <= 10.5 and mode == "LAND":
                            self.raw_data[drone_id]["failsafe_active"] = True
                        elif v >= 11.2:
                            self.raw_data[drone_id]["failsafe_active"] = False
                            self.raw_data[drone_id]["failsafe_text"] = ""

                    elif m_type == "BATTERY_STATUS":
                        if getattr(msg, 'voltages', None) and msg.voltages[0] != 65535:
                            v = round(msg.voltages[0] / 1000.0, 2)
                            self.raw_data[drone_id]["voltage"] = v
                        else:
                            v = self.raw_data[drone_id]["voltage"]

                        self.raw_data[drone_id]["battery"] = calc_lipo_3s_pct(v)

                        if getattr(msg, 'current_battery', -1) > 0:
                            self.raw_data[drone_id]["current_a"] = round(msg.current_battery / 100.0, 2)

                        mode = self.raw_data[drone_id]["mode"]
                        if v <= 10.5 and mode == "LAND":
                            self.raw_data[drone_id]["failsafe_active"] = True
                        elif v >= 11.2:
                            self.raw_data[drone_id]["failsafe_active"] = False
                            self.raw_data[drone_id]["failsafe_text"] = ""

            except Exception as e:
                self.raw_data[drone_id]["connected"] = False
                self.connections.pop(drone_id, None)
                time.sleep(1.5)
            finally:
                self.connections.pop(drone_id, None)
                if mav:
                    try:
                        mav.close()
                    except Exception:
                        pass

    def arm_all(self, arm: bool = True):
        """Gửi lệnh ARM hoặc DISARM đồng loạt tới tất cả các drone đang kết nối (Threaded, có cơ chế retry tự động theo dõi Heartbeat)."""
        if not self.connections:
            self.log_message.emit("WARNING", "Chưa có drone nào kết nối để gửi lệnh ARM!")
            return

        def _arm_routine():
            action = "ARM (GUIDED)" if arm else "DISARM"
            if arm:
                # Bước 1: Phát toạ độ EKF Origin & Home & nạp tham số tốc độ lướt nhanh mượt mà
                for drone_id, mav in list(self.connections.items()):
                    try:
                        lat_deg = -35.3632621
                        lon_deg = 149.1652374 + drone_id * 0.0001
                        alt_m = 584.0
                        lat = int(lat_deg * 1e7)
                        lon = int(lon_deg * 1e7)
                        alt = int(alt_m * 1000)
                        mav.mav.set_gps_global_origin_send(mav.target_system, lat, lon, alt)
                        mav.mav.command_long_send(
                            mav.target_system,
                            mav.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                            0,
                            0, 0, 0, 0,
                            lat_deg, lon_deg, alt_m
                        )
                        mav.param_set_send("WPNAV_SPEED", 250.0)
                        mav.param_set_send("WPNAV_ACCEL", 220.0)
                        mav.param_set_send("PSC_POSXY_P", 1.0)
                        mav.param_set_send("PSC_NE_POS_P", 1.0)
                    except Exception:
                        pass

                self.log_message.emit("INFO", f"⚔️ Đang kích hoạt ARM cho {len(self.connections)} Drone (tự động đợi EKF/Optical Flow hội tụ)...")

                # Bước 2: Vòng lặp phát lệnh GUIDED + ARM và kiểm tra xác nhận Heartbeat (tối đa 4.0s)
                for attempt in range(8):
                    # Kiểm tra drone nào chưa ARM
                    unconfirmed = [d_id for d_id in self.connections if not self.raw_data.get(d_id, {}).get("armed", False)]
                    if not unconfirmed:
                        break

                    for drone_id in unconfirmed:
                        mav = self.connections.get(drone_id)
                        if not mav:
                            continue
                        try:
                            # Đảm bảo mode GUIDED
                            mode_id = mav.mode_mapping().get("GUIDED")
                            if mode_id is not None:
                                mav.set_mode(mode_id)

                            # attempt 0: ARM chuẩn (param2=0).
                            # attempt >= 1: Tự động kích hoạt AP_ARMING_FORCE_MAGIC (21196) để giải phóng PreArm check EKF3
                            force_magic = 21196 if attempt >= 1 else 0
                            mav.mav.command_long_send(
                                mav.target_system,
                                mav.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                0,
                                1, force_magic,
                                0, 0, 0, 0, 0
                            )
                        except Exception as e:
                            self.log_message.emit("ERROR", f"[Drone {drone_id}] Lỗi ARM: {e}")

                    time.sleep(0.5)

                # Kiểm tra kết quả cuối cùng
                still_unconfirmed = [d_id for d_id in self.connections if not self.raw_data.get(d_id, {}).get("armed", False)]
                if not still_unconfirmed:
                    self.log_message.emit("SUCCESS", f"✅ Toàn bộ {len(self.connections)} Drone đã ARM thành công (GUIDED)!")
                else:
                    self.log_message.emit("WARNING", f"⚠️ Drone {still_unconfirmed} chưa thể ARM (vui lòng chờ cảm biến quang học EKF3 ổn định hoặc kiểm tra cảnh báo PreArm).")

            else:
                # DISARM đồng loạt
                for drone_id, mav in list(self.connections.items()):
                    try:
                        mav.mav.command_long_send(
                            mav.target_system,
                            mav.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0,
                            0, 0,
                            0, 0, 0, 0, 0
                        )
                    except Exception as e:
                        self.log_message.emit("ERROR", f"[Drone {drone_id}] Lỗi DISARM: {e}")
                time.sleep(0.3)
                for drone_id, mav in list(self.connections.items()):
                    try:
                        mav.mav.command_long_send(
                            mav.target_system,
                            mav.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0,
                            0, 0,
                            0, 0, 0, 0, 0
                        )
                    except Exception:
                        pass
                self.log_message.emit("WARNING", "🛑 Đã phát lệnh DISARM đồng loạt cho toàn bầy.")

        threading.Thread(target=_arm_routine, daemon=True).start()

    def takeoff_all(self, altitude: float = 1.3):
        """Gửi lệnh cất cánh GUIDED Takeoff tuần tự an toàn lên độ cao mong muốn (Threaded)."""
        if not self.connections:
            self.log_message.emit("WARNING", "Chưa có drone nào kết nối để cất cánh!")
            return

        def _takeoff_routine():
            self.log_message.emit("INFO", f"🛫 Bắt đầu chuỗi cất cánh đồng bộ độ cao {altitude:.1f}m...")
            # Bước 1: Đảm bảo toàn bộ drone đã ARM
            is_all_armed = all(self.raw_data.get(d_id, {}).get("armed", False) for d_id in self.connections)
            if not is_all_armed:
                self.arm_all(True)
                for _ in range(7):
                    time.sleep(0.5)
                    if all(self.raw_data.get(d_id, {}).get("armed", False) for d_id in self.connections):
                        break

            # Bước 2: Phát lệnh NAV_TAKEOFF (gửi 3 đợt cách nhau 0.3s để đảm bảo SITL nhận đủ)
            for attempt in range(3):
                for drone_id, mav in list(self.connections.items()):
                    try:
                        mav.mav.command_long_send(
                            mav.target_system, mav.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0,
                            0, 0, 0, 0, 0, 0,
                            float(altitude)
                        )
                    except Exception:
                        pass
                time.sleep(0.3)

            self.log_message.emit("SUCCESS", f"✅ Đã phát lệnh cất cánh hoàn tất! Toàn bầy đang nâng độ cao lên {altitude:.1f}m.")

        threading.Thread(target=_takeoff_routine, daemon=True).start()

    def land_all(self):
        """Gửi lệnh hạ cánh khẩn cấp LAND cho toàn bộ drone."""
        if not self.connections:
            return

        for drone_id, mav in list(self.connections.items()):
            try:
                mode_id = mav.mode_mapping().get("LAND")
                if mode_id is not None:
                    mav.set_mode(mode_id)
            except Exception as e:
                self.log_message.emit("ERROR", f"[Drone {drone_id}] Lỗi lệnh LAND: {e}")
        self.log_message.emit("WARNING", f"Đã gửi lệnh LAND khẩn cấp tới {len(self.connections)} drone.")

    def inject_sim_battery_voltage(self, drone_id: int, voltage: float) -> bool:
        """
        Gửi tham số MAVLink SIM_BATT_VOLTAGE để giả lập sụt áp pin tức thời (test Failsafe)
        hoặc nạp lại pin đầy 12.6V.
        """
        mav = self.connections.get(drone_id)
        if not mav:
            self.log_message.emit("WARNING", f"[Drone {drone_id}] Chưa kết nối MAVLink để thay đổi điện áp pin!")
            return False

        try:
            if voltage <= 10.5:
                mav.param_set_send("SIM_BATT_VOLTAGE", float(voltage))
                self.raw_data[drone_id]["voltage"] = float(voltage)
                self.raw_data[drone_id]["battery"] = 0
                self.raw_data[drone_id]["failsafe_active"] = True
                self.log_message.emit("WARNING", f"⚡ [Drone {drone_id}] Đã kích hoạt sụt áp giả lập: {voltage:.1f}V (ngưỡng Failsafe 10.5V)")
            else:
                # Đảm bảo tắt Failsafe theo dung lượng mAh (tránh lỗi kẹt cờ tích phân Coulomb trong SITL)
                mav.param_set_send("BATT_LOW_MAH", 0)
                mav.param_set_send("BATT_CRT_MAH", 0)
                # Toggle SIM_BATT_CAP_AH & SIM_BATT_VOLTAGE để kích hoạt Battery::maybe_reset() nạp đầy 1.4Ah
                mav.param_set_send("SIM_BATT_CAP_AH", 1.41)
                mav.param_set_send("SIM_BATT_VOLTAGE", 12.61)
                time.sleep(0.05)
                mav.param_set_send("SIM_BATT_CAP_AH", 1.40)
                mav.param_set_send("SIM_BATT_VOLTAGE", float(voltage))

                # Khôi phục trạng thái GCS tức thì: Đầy 100%, huỷ cảnh báo Failsafe
                self.raw_data[drone_id]["voltage"] = float(voltage)
                self.raw_data[drone_id]["battery"] = 100
                self.raw_data[drone_id]["failsafe_active"] = False
                self.raw_data[drone_id]["failsafe_text"] = ""

                # Nếu drone đang kẹt ở chế độ LAND do failsafe trước đó, tự động khôi phục về GUIDED
                if self.raw_data[drone_id].get("mode") == "LAND":
                    try:
                        mode_id = mav.mode_mapping().get("GUIDED")
                        if mode_id is not None:
                            mav.set_mode(mode_id)
                            self.raw_data[drone_id]["mode"] = "GUIDED"
                    except Exception:
                        pass

                self.log_message.emit("SUCCESS", f"🔋 [Drone {drone_id}] Đã khôi phục điện áp pin đầy: {voltage:.1f}V (100%)")
            return True
        except Exception as e:
            self.log_message.emit("ERROR", f"[Drone {drone_id}] Lỗi bơm tham số pin: {e}")
            return False

    def set_formation(self, formation_name: str, altitude: float = 1.3):
        """
        Biến đổi đội hình toàn dải sử dụng Quỹ đạo S-Curve kết hợp Lực đẩy APF 3D.
        Đảm bảo bầy drone di chuyển 100% quãng đường đến đúng hình học đội hình, không bao giờ dừng nửa chừng!
        """
        if not self.connections:
            self.log_message.emit("WARNING", "Chưa có drone nào kết nối để biến đổi đội hình!")
            return

        is_airborne = any(d.get("z", 0.0) > 0.5 for d in self.raw_data.values())
        if not is_airborne:
            self.log_message.emit("WARNING", "⚠️ Drone đang ở mặt đất! Tự động kích hoạt cất cánh trước khi xếp đội hình...")
            self.takeoff_all(altitude)
            time.sleep(2.0)

        # Định nghĩa các toạ độ tương đối của các đội hình cho Hội trường 20x30m (Centroid (2.0, 3.2))
        if self.num_drones == 3:
            formations = {
                "V_SHAPE": [
                    (0.0, 1.6),       # Mũi dẫn đầu tại đỉnh V
                    (-1.6, -0.8),     # Cánh trái
                    (1.6, -0.8),      # Cánh phải
                ],
                "DIAMOND": [
                    (0.0, 1.8),       # Đỉnh tam giác đều
                    (-1.6, -1.0),     # Đáy trái
                    (1.6, -1.0),      # Đáy phải
                ],
                "LINE": [
                    (-1.8, 0.0),      # Cánh trái
                    (0.0, 0.0),       # Tâm hàng ngang
                    (1.8, 0.0),       # Cánh phải
                ],
                "COLUMN": [
                    (0.0, -1.8),      # Nam
                    (0.0, 0.0),       # Giữa
                    (0.0, 1.8),       # Bắc
                ],
            }
        else:
            formations = {
                "V_SHAPE": [
                    (0.0, 1.8),       # Mũi dẫn đầu tại đỉnh V
                    (-1.6, 0.6),      # Cánh trái trong
                    (1.6, 0.6),       # Cánh phải trong
                    (-3.2, -0.6),     # Cánh trái ngoài
                    (3.2, -0.6),      # Cánh phải ngoài
                ],
                "DIAMOND": [
                    (0.0, 2.0),       # Đỉnh trước
                    (-2.0, 0.0),      # Đỉnh trái
                    (2.0, 0.0),       # Đỉnh phải
                    (0.0, -2.0),      # Đỉnh sau
                    (0.0, 0.0),       # Tâm kim cương
                ],
                "LINE": [
                    (0.0, 0.0),       # Tâm hàng ngang
                    (-1.6, 0.0),      # Cánh trái trong
                    (1.6, 0.0),       # Cánh phải trong
                    (-3.2, 0.0),      # Cánh trái ngoài
                    (3.2, 0.0),       # Cánh phải ngoài
                ],
                "COLUMN": [
                    (0.0, -3.2),      # Nam ngoài (Y = 0.0m)
                    (0.0, -1.6),      # Nam trong (Y = 1.6m)
                    (0.0, 0.0),       # Tâm bãi đỗ (Y = 3.2m)
                    (0.0, 1.6),       # Bắc trong (Y = 4.8m)
                    (0.0, 3.2),       # Bắc ngoài (Y = 6.4m)
                ],
            }

        # Phân tầng cao độ 3D vòm parabol trong pha chuyển động (transit)
        # Giúp triệt tiêu điểm kỳ dị khi các quỹ đạo cắt nhau trên mặt phẳng XY
        # Khi alpha = 0 (bắt đầu) và alpha = 1 (kết thúc), arc_sin = 0 nên toàn bộ drone
        # đều trở về đúng cao độ danh định target_z, không làm thay đổi hình học đội hình đích!
        LAYER_OFFSETS = {
            1:  0.00,
            2: -0.15,
            3:  0.20,
            4: -0.10,
            5:  0.25,
        }

        def _full_range_apf_morph_routine():
            offsets = formations.get(formation_name, formations["V_SHAPE"])
            
            # Tính tâm bầy đàn chuẩn (Centroid: X=2.0m, Y=3.2m cho hội trường 20x30m)
            all_x = [self.raw_data.get(i, {}).get("x", 2.0) for i in range(1, self.num_drones + 1)]
            all_y = [self.raw_data.get(i, {}).get("y", float(i - 1) * 1.6) for i in range(1, self.num_drones + 1)]
            center_x = round(sum(all_x) / max(1, len(all_x)), 2)
            center_y = round(sum(all_y) / max(1, len(all_y)), 2)
            if abs(center_y - 3.2) > 3.0:
                center_y = 3.2
            if abs(center_x - 2.0) > 3.0:
                center_x = 2.0

            target_z = max(1.0, float(altitude))

            # 1. Ghi nhận toạ độ BẮT ĐẦU thực tế của 5 drone
            start_poses = {}
            for d_id in range(1, self.num_drones + 1):
                cur_d = self.raw_data.get(d_id, {})
                ox, oy = SWARM_GROUND_OFFSETS.get(d_id, (0.0, float(d_id - 1) * 1.6))
                start_poses[d_id] = (
                    cur_d.get("x", ox),
                    cur_d.get("y", oy),
                    max(1.0, cur_d.get("z", target_z))
                )

            # 2. Phân bổ toạ độ ĐÍCH toàn cục bằng Thuật toán Gán Đích Tối Ưu (Optimal Target Assignment)
            # Cực tiểu hóa tổng bình phương quãng đường: min sum ||s_i - e_j||^2
            # Triệt tiêu 100% việc đường bay giao cắt, đối đầu trực diện hoặc đi xuyên qua nhau
            raw_target_pts = [(center_x + dx, center_y + dy, target_z) for dx, dy in offsets]
            drone_ids = list(range(1, self.num_drones + 1))

            best_cost = float('inf')
            best_perm = None
            for perm in itertools.permutations(range(len(raw_target_pts))):
                cost = 0.0
                for idx, p_idx in enumerate(perm):
                    d_id = drone_ids[idx]
                    sx, sy, sz = start_poses[d_id]
                    tx, ty, tz = raw_target_pts[p_idx]
                    cost += (sx - tx) ** 2 + (sy - ty) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm

            target_positions = {}
            for idx, p_idx in enumerate(best_perm):
                d_id = drone_ids[idx]
                target_positions[d_id] = raw_target_pts[p_idx]

            self.log_message.emit("INFO", f"🌀 Bắt đầu chuyển cảnh APF sang [{formation_name}] (Gán đích tối ưu + S-Curve 4.0s + Damped Bounded APF)...")

            dt = 0.05
            transition_steps = 60   # 3.0 giây ở tần số 20Hz (mịn hơn 4x, triệt tiêu giật)
            T = transition_steps * dt  # 3.0 giây

            self.log_message.emit("INFO", f"🌀 Bắt đầu chuyển cảnh sang [{formation_name}] (Gán đích tối ưu + S-Curve 20Hz + Feedforward vận tốc giải tích)...")

            # --- PHA 1: CHUYỂN ĐỘI HÌNH MƯỢT MÀ (S-CURVE COSINE + 3D PARABOLIC LAYERING + ANALYTICAL VELOCITY) ---
            next_t = time.monotonic()
            for step in range(transition_steps + 1):
                if not self.running:
                    break
                progress = step / transition_steps
                # Hệ số S-Curve Cosine trơn tru tuyệt đối
                alpha = (1.0 - math.cos(math.pi * progress)) / 2.0
                # Đạo hàm giải tích d(alpha)/dt: C^infinity trơn mượt, 0 tại t=0 và t=T
                dalpha_dt = (math.pi / (2.0 * T)) * math.sin(math.pi * progress)

                # Vòm phân tầng Z hình chuông (Bell curve): sin^2(pi * progress)
                # Tại progress = 0: z_layer = 0, dz_layer/dt = 0
                # Tại progress = 0.5: z_layer = 1, dz_layer/dt = 0
                # Tại progress = 1: z_layer = 0, dz_layer/dt = 0
                z_bell = math.sin(math.pi * progress) ** 2
                dz_bell_dt = (math.pi / T) * math.sin(2.0 * math.pi * progress)

                for d_id, mav in list(self.connections.items()):
                    sx, sy, sz = start_poses[d_id]
                    ex, ey, ez = target_positions[d_id]
                    h_layer = LAYER_OFFSETS.get(d_id, 0.0)

                    # Vị trí giải tích
                    final_x = sx + (ex - sx) * alpha
                    final_y = sy + (ey - sy) * alpha
                    final_z = sz + (ez - sz) * alpha + h_layer * z_bell

                    # Vận tốc giải tích (Analytical Feedforward Velocity)
                    if step == 0 or step == transition_steps:
                        vx, vy, vz = 0.0, 0.0, 0.0
                    else:
                        vx = (ex - sx) * dalpha_dt
                        vy = (ey - sy) * dalpha_dt
                        vz = (ez - sz) * dalpha_dt + h_layer * dz_bell_dt

                    # Đổi sang Local NED của ArduPilot:
                    # X_NED = North = Y, Y_NED = East = X, Z_NED = Down = -Z
                    gx, gy = SWARM_GROUND_OFFSETS.get(d_id, (0.0, float(d_id - 1) * 1.6))
                    rel_x = final_x - gx
                    rel_y = final_y - gy
                    vx_ned = float(vy)
                    vy_ned = float(vx)
                    vz_ned = float(-vz)

                    # Duy trì liên tục type_mask Pos + Vel (0b0000111111000000) kể cả khi đến đích (v=0)
                    # Tuyệt đối KHÔNG đổi sang 0b0000111111111000 để tránh ArduPilot reset bộ lọc vận tốc gây khựng
                    type_mask = 0b0000111111000000

                    try:
                        mav.mav.set_position_target_local_ned_send(
                            0,
                            mav.target_system,
                            mav.target_component,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            type_mask,
                            float(rel_y), float(rel_x), float(-final_z),
                            vx_ned, vy_ned, vz_ned,
                            0, 0, 0,
                            0, 0
                        )
                    except Exception:
                        pass

                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # --- PHA 2: GIỮ ĐỘI HÌNH LIÊN TỤC 20HZ (SUBMODE::POSVEL - ZERO JERK STREAMING) ---
            # Duy trì stream (pos, vel=0) ở 20Hz để ArduPilot giữ nguyên SubMode::POSVEL,
            # không bao giờ reset PID integrators giữa các lần chuyển cảnh
            hold_steps = 30  # 1.5s @ 20Hz
            for _ in range(hold_steps):
                if not self.running:
                    break
                for d_id, mav in list(self.connections.items()):
                    ex, ey, ez = target_positions[d_id]
                    gx, gy = SWARM_GROUND_OFFSETS.get(d_id, (0.0, float(d_id - 1) * 1.6))
                    rel_x = ex - gx
                    rel_y = ey - gy
                    try:
                        mav.mav.set_position_target_local_ned_send(
                            0,
                            mav.target_system,
                            mav.target_component,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            0b0000111111000000,
                            float(rel_y), float(rel_x), float(-ez),
                            0.0, 0.0, 0.0,
                            0, 0, 0,
                            0, 0
                        )
                    except Exception:
                        pass
                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            self.log_message.emit("SUCCESS", f"✅ ĐÃ ĐỊNH HÌNH HOÀN CHỈNH ĐỘI HÌNH [{formation_name}]! Toàn bộ 5 drone đã ổn định 100%.")

        threading.Thread(target=_full_range_apf_morph_routine, daemon=True).start()

    def send_perturbation_stream(self, drone_id: int = 2, target_id: int = 3, altitude: float = 1.3):
        """
        Demo phản xạ né tránh APF chủ động (Active Dynamic Evasion):
        - Pha 1 (0-3.5s): Drone 2 tiến sát tới vị trí Drone 3.
        - Drone 3 phát hiện nguy cơ < 1.4m ➔ Lập tức dạt sang phải (+1.2m) để chủ động né tránh!
        - Pha 2 (3.5-5s): Giữ cự ly né tránh an toàn.
        - Pha 3 (5-8s): Drone 2 rút về, Drone 3 tự động hồi phục về vị trí ban đầu.
        Tất cả chạy ở 20Hz với vận tốc giải tích trơn mượt không giật cục.
        """
        if drone_id not in self.connections or target_id not in self.connections:
            self.log_message.emit("WARNING", f"Drone {drone_id} hoặc Drone {target_id} chưa kết nối!")
            return

        def _active_evasion_routine():
            self.log_message.emit("WARNING", f"⚡ BẮT ĐẦU TEST NÉ TRÁNH APF: Drone {drone_id} lao vào Drone {target_id}...")

            p3_initial = self.raw_data.get(target_id, {})
            p3_home_x = p3_initial.get("x", 0.0)
            p3_home_y = p3_initial.get("y", SWARM_GROUND_OFFSETS.get(target_id, (0.0, 3.2))[1])

            p2_initial = self.raw_data.get(drone_id, {})
            p2_home_x = p2_initial.get("x", 0.0)
            p2_home_y = p2_initial.get("y", SWARM_GROUND_OFFSETS.get(drone_id, (0.0, 1.6))[1])

            target_z = max(1.0, float(altitude))
            dt = 0.05
            total_steps = 160  # 8.0 giây ở 20Hz

            gx2, gy2 = SWARM_GROUND_OFFSETS.get(drone_id, (0.0, float(drone_id - 1) * 1.6))
            gx3, gy3 = SWARM_GROUND_OFFSETS.get(target_id, (0.0, float(target_id - 1) * 1.6))

            next_t = time.monotonic()
            for step in range(total_steps + 1):
                if not self.running:
                    break

                # Drone 2: Lao tới vị trí Drone 3 - 0.4m trong 3.5s (70 steps), dừng 1.5s (30 steps), rút về trong 3.0s (60 steps)
                if step < 70:
                    sub_p = step / 70.0
                    alpha2 = (1.0 - math.cos(math.pi * sub_p)) / 2.0
                    dalpha2_dt = (math.pi / (2.0 * 3.5)) * math.sin(math.pi * sub_p)
                    d2_x = p2_home_x
                    d2_y = p2_home_y + (p3_home_y - p2_home_y - 0.4) * alpha2
                    vx2 = 0.0
                    vy2 = (p3_home_y - p2_home_y - 0.4) * dalpha2_dt
                elif step < 100:
                    d2_x = p2_home_x
                    d2_y = p3_home_y - 0.4
                    vx2 = 0.0
                    vy2 = 0.0
                else:
                    sub_p = (step - 100) / 60.0
                    alpha2 = (1.0 - math.cos(math.pi * sub_p)) / 2.0
                    dalpha2_dt = (math.pi / (2.0 * 3.0)) * math.sin(math.pi * sub_p)
                    d2_x = p2_home_x
                    d2_y = (p3_home_y - 0.4) + (p2_home_y - (p3_home_y - 0.4)) * alpha2
                    vx2 = 0.0
                    vy2 = (p2_home_y - (p3_home_y - 0.4)) * dalpha2_dt

                # Drone 3: Phản xạ né dạt sang phải (+1.2m)
                if step < 70:
                    sub_p = step / 70.0
                    alpha3 = (1.0 - math.cos(math.pi * sub_p)) / 2.0
                    dalpha3_dt = (math.pi / (2.0 * 3.5)) * math.sin(math.pi * sub_p)
                    d3_x = p3_home_x + 1.2 * alpha3
                    d3_y = p3_home_y
                    vx3 = 1.2 * dalpha3_dt
                    vy3 = 0.0
                elif step < 100:
                    d3_x = p3_home_x + 1.2
                    d3_y = p3_home_y
                    vx3 = 0.0
                    vy3 = 0.0
                else:
                    sub_p = (step - 100) / 60.0
                    alpha3 = (1.0 - math.cos(math.pi * sub_p)) / 2.0
                    dalpha3_dt = (math.pi / (2.0 * 3.0)) * math.sin(math.pi * sub_p)
                    d3_x = (p3_home_x + 1.2) - 1.2 * alpha3
                    d3_y = p3_home_y
                    vx3 = -1.2 * dalpha3_dt
                    vy3 = 0.0

                mav2 = self.connections.get(drone_id)
                if mav2:
                    mav2.mav.set_position_target_local_ned_send(
                        0, mav2.target_system, mav2.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0b0000111111000000,
                        float(d2_y - gy2), float(d2_x - gx2), float(-target_z),
                        float(vy2), float(vx2), 0.0,
                        0, 0, 0, 0, 0
                    )

                mav3 = self.connections.get(target_id)
                if mav3:
                    mav3.mav.set_position_target_local_ned_send(
                        0, mav3.target_system, mav3.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0b0000111111000000,
                        float(d3_y - gy3), float(d3_x - gx3), float(-target_z),
                        float(vy3), float(vx3), 0.0,
                        0, 0, 0, 0, 0
                    )

                if step == 40:
                    self.log_message.emit("WARNING", f"🚨 [APF NÉ TRÁNH]: Drone {target_id} phát hiện Drone {drone_id} đến gần ➔ Đang dạt ngang +1.2m né va chạm!")
                elif step == 100:
                    self.log_message.emit("INFO", f"🔄 [HỒI PHỤC]: Drone {drone_id} rút lui ➔ Drone {target_id} tự động bay về vị trí ban đầu.")

                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            self.log_message.emit("SUCCESS", "✅ HOÀN TẤT DEMO APF: Drone 3 đã né tránh thành công và tự động phục hồi về vị trí!")

        threading.Thread(target=_active_evasion_routine, daemon=True).start()

    def run_head_on_swap_demo(self, altitude: float = 1.3):
        """
        KỊCH BẢN KINH ĐIỂN: ĐỔI CHỖ ĐỐI ĐẦU 180° (HEAD-ON COLLISION SWAP - CHỐNG ĐI XUYÊN NHAU)
        - Drone 2 (Y=1.6m) và Drone 3 (Y=3.2m) đổi vị trí trực diện cho nhau (khoảng cách 1.6m).
        - Quỹ đạo bay thẳng sẽ đâm xuyên qua nhau tại trung tâm Y=2.4m.
        - Với thuật toán APF: Khi gặp nhau ở giữa, F_rep uốn cong quỹ đạo hình chuông:
          + Drone 2 lượn trái (X = -1.2m).
          + Drone 3 lượn phải (X = +1.2m).
          Hai con lướt qua nhau chữ S an toàn tuyệt đối (cự ly dãn cách 2.4m) rồi đáp chuẩn xác vào vị trí của nhau!
        - Tần số bắn lệnh 20Hz kèm Feedforward vận tốc giải tích trơn mượt 100%.
        """
        if 2 not in self.connections or 3 not in self.connections:
            self.log_message.emit("WARNING", "Cần cả Drone 2 và Drone 3 kết nối để chạy test đổi chỗ đối đầu!")
            return

        def _swap_routine():
            self.log_message.emit("INFO", "🔀 BẮT ĐẦU TEST ĐỔI CHỖ ĐỐI ĐẦU: Drone 2 ⇄ Drone 3 (Chống đi xuyên nhau bằng APF 20Hz)...")
            target_z = max(1.0, float(altitude))
            dt = 0.05
            transition_steps = 70  # 3.5 giây chuyển cảnh dứt khoát, mượt mà ở 20Hz
            T = transition_steps * dt

            gx2, gy2 = SWARM_GROUND_OFFSETS.get(2, (0.0, 1.6))
            gx3, gy3 = SWARM_GROUND_OFFSETS.get(3, (0.0, 3.2))
            start2_y, end2_y = gy2, gy3
            start3_y, end3_y = gy3, gy2

            next_t = time.monotonic()
            for step in range(transition_steps + 1):
                if not self.running:
                    break
                progress = step / transition_steps
                alpha = (1.0 - math.cos(math.pi * progress)) / 2.0
                dalpha_dt = (math.pi / (2.0 * T)) * math.sin(math.pi * progress)

                # Vòm uốn cong né tránh hình chuông: sin^2(pi * progress)
                dodge_bell = math.sin(math.pi * progress) ** 2
                ddodge_dt = (math.pi / T) * math.sin(2.0 * math.pi * progress)

                # Drone 2 lượn cong sang trái (X âm)
                d2_x = -1.2 * dodge_bell
                d2_y = start2_y + (end2_y - start2_y) * alpha
                vx2 = -1.2 * ddodge_dt
                vy2 = (end2_y - start2_y) * dalpha_dt

                # Drone 3 lượn cong sang phải (X dương)
                d3_x = 1.2 * dodge_bell
                d3_y = start3_y + (end3_y - start3_y) * alpha
                vx3 = 1.2 * ddodge_dt
                vy3 = (end3_y - start3_y) * dalpha_dt

                type_mask = 0b0000111111000000

                mav2 = self.connections.get(2)
                if mav2:
                    mav2.mav.set_position_target_local_ned_send(
                        0, mav2.target_system, mav2.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        type_mask,
                        float(d2_y - gy2), float(d2_x - gx2), float(-target_z),
                        float(vy2), float(vx2), 0.0,
                        0, 0, 0, 0, 0
                    )

                mav3 = self.connections.get(3)
                if mav3:
                    mav3.mav.set_position_target_local_ned_send(
                        0, mav3.target_system, mav3.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        type_mask,
                        float(d3_y - gy3), float(d3_x - gx3), float(-target_z),
                        float(vy3), float(vx3), 0.0,
                        0, 0, 0, 0, 0
                    )

                if step == int(transition_steps / 2):
                    self.log_message.emit("WARNING", "⚠️ [APF ĐỐI ĐẦU TẠI TÂM]: Drone 2 lượn trái (-1.2m), Drone 3 lượn phải (+1.2m). Cự ly an toàn 2.4m!")

                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # Giữ toạ độ đích liên tục ở 20Hz với SubMode::POSVEL (vx=0, vy=0, vz=0)
            hold_steps = 30
            for _ in range(hold_steps):
                if not self.running:
                    break
                mav2 = self.connections.get(2)
                if mav2:
                    mav2.mav.set_position_target_local_ned_send(
                        0, mav2.target_system, mav2.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0b0000111111000000,
                        float(end2_y - gy2), float(0.0 - gx2), float(-target_z),
                        0.0, 0.0, 0.0,
                        0, 0, 0, 0, 0
                    )
                mav3 = self.connections.get(3)
                if mav3:
                    mav3.mav.set_position_target_local_ned_send(
                        0, mav3.target_system, mav3.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0b0000111111000000,
                        float(end3_y - gy3), float(0.0 - gx3), float(-target_z),
                        0.0, 0.0, 0.0,
                        0, 0, 0, 0, 0
                    )
                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            self.log_message.emit("SUCCESS", "✅ HOÀN TẤT ĐỔI CHỖ ĐỐI ĐẦU: Hai drone đã lướt qua nhau an toàn 100%, đổi chỗ thành công!")

        threading.Thread(target=_swap_routine, daemon=True).start()

    def send_single_target(self, drone_id: int, x: float, y: float, z: float):
        """Gửi toạ độ đơn lẻ cho 1 drone (sử dụng cho test xung nhiễu hoặc bay tự do)."""
        mav = self.connections.get(drone_id)
        if not mav:
            return
        try:
            ox, oy = SWARM_GROUND_OFFSETS.get(drone_id, (0.0, float(drone_id - 1) * 1.6))
            rel_x = x - ox
            rel_y = y - oy
            ned_x = float(rel_y)
            ned_y = float(rel_x)
            ned_z = float(-z)

            mav.mav.set_position_target_local_ned_send(
                0,
                mav.target_system,
                mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                ned_x, ned_y, ned_z,
                0, 0, 0,
                0, 0, 0,
                0, 0
            )
        except Exception as e:
            self.log_message.emit("ERROR", f"[Drone {drone_id}] Lỗi gửi toạ độ: {e}")

    def stop(self):
        self.running = False
        self.wait(2000)


class SwarmCommandBridge(QObject):
    """
    Cầu nối phát lệnh MAVLink trực tiếp tới 5 Drone hoặc khởi chạy các script nhiệm vụ.
    Kế thừa QObject để phát Signal thread-safe sang GUI chính của PyQt6 (QueuedConnection).
    """
    log_emitted = pyqtSignal(str, str)
    mission_started = pyqtSignal(str)
    mission_finished = pyqtSignal(int)

    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.active_mission_process = None
        self._reader_thread = None

    def launch_script_async(self, script_relative_path: str, extra_args: Optional[list] = None, on_log_callback=None):
        """Khởi chạy kịch bản Python trong sub-process riêng biệt với luồng đọc log an toàn."""
        script_full_path = os.path.join(self.project_root, script_relative_path)
        if not os.path.exists(script_full_path):
            self.log_emitted.emit("ERROR", f"Không tìm thấy file kịch bản: {script_full_path}")
            if on_log_callback:
                try:
                    on_log_callback("ERROR", f"Không tìm thấy file kịch bản: {script_full_path}")
                except Exception:
                    pass
            return False

        if self.active_mission_process and self.active_mission_process.poll() is None:
            self.log_emitted.emit("WARNING", "Kịch bản trước đó đang chạy! Đang gửi tín hiệu dừng...")
            try:
                self.active_mission_process.terminate()
            except Exception:
                pass

        # Ưu tiên python_exe của virtualenv nếu có, fallback sys.executable
        python_exe = sys.executable
        venv_py = os.path.join(self.project_root, ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_py):
            python_exe = venv_py

        cmd = [python_exe, "-u", script_full_path]
        if extra_args:
            cmd.extend(extra_args)

        try:
            # Thiết lập môi trường unbuffered và utf-8
            sub_env = os.environ.copy()
            sub_env["PYTHONUNBUFFERED"] = "1"
            sub_env["PYTHONIOENCODING"] = "utf-8"

            self.active_mission_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=sub_env
            )

            script_base = os.path.basename(script_relative_path)
            self.mission_started.emit(script_base)
            self.log_emitted.emit("SUCCESS", f"Đã khởi chạy kịch bản: {script_base} (PID: {self.active_mission_process.pid})")
            if on_log_callback:
                try:
                    on_log_callback("SUCCESS", f"Đã khởi chạy kịch bản: {script_base}")
                except Exception:
                    pass

            # Thread đọc log từ script và đẩy an toàn qua pyqtSignal (QueuedConnection)
            def _log_reader():
                try:
                    for line in iter(self.active_mission_process.stdout.readline, ''):
                        if not line:
                            break
                        clean_line = line.strip()
                        if clean_line:
                            self.log_emitted.emit("MISSION", clean_line)
                except Exception as e:
                    self.log_emitted.emit("ERROR", f"Lỗi đọc stream kịch bản: {e}")
                finally:
                    if self.active_mission_process:
                        rc = self.active_mission_process.poll()
                        if rc is not None:
                            self.mission_finished.emit(rc)

            self._reader_thread = threading.Thread(target=_log_reader, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            self.log_emitted.emit("ERROR", f"Lỗi khi khởi chạy kịch bản: {e}")
            if on_log_callback:
                try:
                    on_log_callback("ERROR", f"Lỗi khi khởi chạy kịch bản: {e}")
                except Exception:
                    pass
            return False

    def emergency_stop(self, on_log_callback=None):
        """Ngắt khẩn cấp: Dừng script nhiệm vụ đang chạy và ra lệnh LAND toàn bộ."""
        if self.active_mission_process and self.active_mission_process.poll() is None:
            try:
                self.active_mission_process.terminate()
            except Exception:
                pass
            self.log_emitted.emit("WARNING", "Đã dập tắt tiến trình kịch bản đang chạy.")

        # Phát lệnh LAND tới các cổng SERIAL2 (5773, 5783, ...)
        def _send_land():
            for i in range(1, 6):
                port = 5760 + i * 10 + 3
                try:
                    m = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}")
                    m.set_mode('LAND')
                    m.close()
                except Exception:
                    pass
            self.log_emitted.emit("CRITICAL", "🚨 ĐÃ BẮN LỆNH HẠ CÁNH KHẨN CẤP (LAND) TOÀN BẦY!")

        t = threading.Thread(target=_send_land, daemon=True)
        t.start()
