#!/usr/bin/env python3
"""
DRONE SWARM GROUND CONTROL STATION (GCS)
Desktop Application xây dựng trên PyQt6
Dành cho hệ thống Swarm 5 Drone ArduPilot (SITL & ESP32-S3 Hardware).
Giao diện Pure Black Tactical (#000000) - Tối giản quân sự, loại bỏ phong cách AI màu mè.
Tích hợp sẵn Bảng Quản Lý Mô Phỏng 1-Click (SITL, 3D Mesh Publisher, RViz2).
"""

import os
import sys
import time
from typing import Dict

# Hỗ trợ font tiếng Việt cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Tắt cảnh báo font DirectWrite bitmap không cần thiết trên Windows
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

# Ép xung nhịp hẹn giờ Windows Scheduler xuống chính xác 1.0ms (triệt tiêu timer jitter 15.6ms của Windows)
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

# Thêm đường dẫn project root vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QTextEdit, QFrame,
    QSplitter, QGridLayout, QScrollArea, QProgressBar
)

from app_gcs.styles import MAIN_STYLESHEET, DRONE_COLORS, DEFAULT_NUM_DRONES
from app_gcs.radar_widget import TacticalRadarWidget
from app_gcs.telemetry_worker import SwarmTelemetryWorker, SwarmCommandBridge, calc_lipo_3s_pct
from app_gcs.sim_manager import SwarmSimulationManager


class DroneCard(QFrame):
    """Thẻ hiển thị thông số Telemetry thời gian thực cho 1 Drone (Pure Black Tactical)."""
    def __init__(self, drone_id: int, parent=None):
        super().__init__(parent)
        self.drone_id = drone_id
        color_info = DRONE_COLORS.get(drone_id, {"hex": "#E0E0E0", "name": "Drone"})
        self.drone_hex = color_info["hex"]
        self.accumulated_flight_sec = 0.0
        self.arm_start_time = None
        self.prev_armed = False

        self._last_connected = None
        self._last_failsafe = None
        self._last_mode = None
        self._last_armed = None
        self._last_elapsed_state = None
        self._last_bat_tier = None
        self._last_thr_tier = None

        self.setObjectName("droneCard")
        self.setStyleSheet(f"""
            QFrame#droneCard {{
                background-color: #0A0A0A;
                border: 1px solid #1E1E1E;
                border-left: 3px solid {self.drone_hex};
                border-radius: 2px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)

        # Header: Tên drone + Chế độ bay
        h_box = QHBoxLayout()
        self.lbl_title = QLabel(f"DRONE {drone_id}" + (" [LEADER]" if drone_id == 1 else ""))
        self.lbl_title.setStyleSheet(f"font-weight: bold; color: {self.drone_hex}; font-size: 12px; font-family: 'Consolas', monospace;")
        self.lbl_mode = QLabel("OFFLINE")
        self.lbl_mode.setStyleSheet("background-color: #141414; color: #666666; padding: 1px 5px; border-radius: 2px; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
        h_box.addWidget(self.lbl_title)
        h_box.addStretch()
        h_box.addWidget(self.lbl_mode)
        layout.addLayout(h_box)

        # Thanh Máu (HP / Battery Health Bar) phong cách Tactical HUD
        hp_layout = QHBoxLayout()
        hp_layout.setSpacing(6)

        self.lbl_hp_icon = QLabel("HP")
        self.lbl_hp_icon.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; color: #777777;")
        hp_layout.addWidget(self.lbl_hp_icon)

        self.pbar_bat = QProgressBar()
        self.pbar_bat.setRange(0, 100)
        self.pbar_bat.setValue(0)
        self.pbar_bat.setTextVisible(False)
        self.pbar_bat.setFixedHeight(8)
        self.pbar_bat.setStyleSheet("""
            QProgressBar {
                background-color: #121212;
                border: 1px solid #1E1E1E;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #333333;
                border-radius: 2px;
            }
        """)
        hp_layout.addWidget(self.pbar_bat, stretch=1)

        self.lbl_bat = QLabel("--V (0%)")
        self.lbl_bat.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; color: #555555;")
        hp_layout.addWidget(self.lbl_bat)
        layout.addLayout(hp_layout)

        # Lưới thông số đối xứng: 4 hàng x 2 cột dữ liệu
        grid = QGridLayout()
        grid.setSpacing(2)

        # Hàng 0: Cao độ & Vận tốc
        grid.addWidget(self._make_label("Cao độ:"), 0, 0)
        self.lbl_alt = QLabel("0.0 m")
        self.lbl_alt.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #DDDDDD;")
        grid.addWidget(self.lbl_alt, 0, 1)

        grid.addWidget(self._make_label("Vận tốc:"), 0, 2)
        self.lbl_spd = QLabel("0.0 m/s")
        self.lbl_spd.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #DDDDDD;")
        grid.addWidget(self.lbl_spd, 0, 3)

        # Hàng 1: Timing thực tế (Đã bay) & Dự kiến (Còn lại)
        grid.addWidget(self._make_label("Đã bay:"), 1, 0)
        self.lbl_elapsed = QLabel("00:00")
        self.lbl_elapsed.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #777777;")
        grid.addWidget(self.lbl_elapsed, 1, 1)

        grid.addWidget(self._make_label("Còn lại:"), 1, 2)
        self.lbl_time = QLabel("--")
        self.lbl_time.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #38BDF8;")
        grid.addWidget(self.lbl_time, 1, 3)

        # Hàng 2: Trạng thái & Dòng xả thực tế (SITL / Hardware)
        grid.addWidget(self._make_label("Trạng thái:"), 2, 0)
        self.lbl_arm = QLabel("DISCONN")
        self.lbl_arm.setStyleSheet("color: #555555; font-family: 'Consolas', monospace; font-size: 10px;")
        grid.addWidget(self.lbl_arm, 2, 1)

        grid.addWidget(self._make_label("Dòng xả:"), 2, 2)
        self.lbl_curr = QLabel("--")
        self.lbl_curr.setStyleSheet("color: #777777; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
        grid.addWidget(self.lbl_curr, 2, 3)

        # Hàng 3: Toạ độ X, Y & Mức ga thực tế
        grid.addWidget(self._make_label("Toạ độ:"), 3, 0)
        self.lbl_pos = QLabel("(0.0m, 0.0m)")
        self.lbl_pos.setStyleSheet("color: #AAAAAA; font-size: 10px; font-family: 'Consolas', monospace;")
        grid.addWidget(self.lbl_pos, 3, 1)

        grid.addWidget(self._make_label("Mức ga:"), 3, 2)
        self.lbl_thr = QLabel("0%")
        self.lbl_thr.setStyleSheet("color: #777777; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
        grid.addWidget(self.lbl_thr, 3, 3)

        layout.addLayout(grid)

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #666666; font-size: 10px;")
        return lbl

    def update_data(self, data: dict):
        connected = data.get("connected", False)
        if not connected:
            if self._last_connected is not False:
                self._last_connected = False
                self._last_failsafe = None
                self._last_mode = None
                self._last_armed = None
                self._last_elapsed_state = None
                self._last_bat_tier = None
                self._last_thr_tier = None

                self.lbl_mode.setText("OFFLINE")
                self.lbl_mode.setStyleSheet("background-color: #141414; color: #666666; padding: 1px 5px; border: 1px solid #222222; font-family: 'Consolas', monospace;")
                self.lbl_arm.setText("DISCONN")
                self.lbl_arm.setStyleSheet("color: #555555; font-family: 'Consolas', monospace;")
                self.lbl_time.setText("--")
                self.lbl_curr.setText("--")
                self.lbl_thr.setText("--")
                self.lbl_thr.setStyleSheet("color: #777777; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
                self.lbl_bat.setText("--V (0%)")
                self.lbl_bat.setStyleSheet("color: #555555; font-family: 'Consolas', monospace; font-size: 10px;")
                self.lbl_hp_icon.setStyleSheet("color: #444444; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold;")
                self.pbar_bat.setValue(0)
                self.pbar_bat.setStyleSheet("""
                    QProgressBar {
                        background-color: #121212;
                        border: 1px solid #1E1E1E;
                        border-radius: 3px;
                    }
                    QProgressBar::chunk {
                        background-color: #333333;
                        border-radius: 2px;
                    }
                """)
                self.setStyleSheet(f"""
                    QFrame#droneCard {{
                        background-color: #0A0A0A;
                        border: 1px solid #1E1E1E;
                        border-left: 3px solid {self.drone_hex};
                        border-radius: 2px;
                    }}
                """)
                base_title = f"DRONE {self.drone_id}" + (" [LEADER]" if self.drone_id == 1 else "")
                self.lbl_title.setText(base_title)
                self.lbl_title.setStyleSheet(f"font-weight: bold; color: {self.drone_hex}; font-size: 12px; font-family: 'Consolas', monospace;")

            if self.arm_start_time is not None:
                self.accumulated_flight_sec += (time.time() - self.arm_start_time)
                self.arm_start_time = None
                self.prev_armed = False
            total_sec = int(self.accumulated_flight_sec)
            if total_sec > 0:
                mm, ss = divmod(total_sec, 60)
                self.lbl_elapsed.setText(f"{mm:02d}:{ss:02d}")
            else:
                self.lbl_elapsed.setText("--:--")
            return

        if self._last_connected is not True:
            self._last_connected = True
            # Đánh dấu nạp lại stylesheet lần đầu sau khi kết nối
            self._last_failsafe = None
            self._last_mode = None
            self._last_armed = None
            self._last_elapsed_state = None
            self._last_bat_tier = None
            self._last_thr_tier = None

        mode = data.get("mode", "UNKNOWN")
        v = data.get("voltage", 12.6)
        is_failsafe = bool(data.get("failsafe_active", False) or (v <= 10.5 and mode == "LAND"))

        # 1. Chỉ áp dụng CSS Thẻ & Tiêu đề khi trạng thái Failsafe thực sự thay đổi
        if is_failsafe != self._last_failsafe:
            self._last_failsafe = is_failsafe
            if is_failsafe:
                self.setStyleSheet("""
                    QFrame#droneCard {
                        background-color: #1A0707;
                        border: 1.5px solid #EF4444;
                        border-left: 4px solid #EF4444;
                        border-radius: 2px;
                    }
                """)
                self.lbl_title.setText(f"DRONE {self.drone_id} [FAILSAFE: LAND]")
                self.lbl_title.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 11px; font-family: 'Consolas', monospace;")
            else:
                self.setStyleSheet(f"""
                    QFrame#droneCard {{
                        background-color: #0A0A0A;
                        border: 1px solid #1E1E1E;
                        border-left: 3px solid {self.drone_hex};
                        border-radius: 2px;
                    }}
                """)
                base_title = f"DRONE {self.drone_id}" + (" [LEADER]" if self.drone_id == 1 else "")
                self.lbl_title.setText(base_title)
                self.lbl_title.setStyleSheet(f"font-weight: bold; color: {self.drone_hex}; font-size: 12px; font-family: 'Consolas', monospace;")
            self._last_mode = None  # Cưỡng chế cập nhật lại style mode

        # 2. Cập nhật nhãn & CSS Mode chỉ khi Mode thực sự thay đổi
        if mode != self._last_mode:
            self._last_mode = mode
            self.lbl_mode.setText(mode)
            if is_failsafe:
                self.lbl_mode.setStyleSheet("background-color: #450A0A; color: #FCA5A5; border: 1px solid #DC2626; padding: 1px 5px; font-weight: bold; font-family: 'Consolas', monospace;")
            elif mode in ["GUIDED", "AUTO"]:
                self.lbl_mode.setStyleSheet("background-color: #0F2D18; color: #4ADE80; border: 1px solid #166534; padding: 1px 5px; font-weight: bold; font-family: 'Consolas', monospace;")
            else:
                self.lbl_mode.setStyleSheet("background-color: #181818; color: #CCCCCC; border: 1px solid #333333; padding: 1px 5px; font-family: 'Consolas', monospace;")
        else:
            self.lbl_mode.setText(mode)

        # 3. Cập nhật nhãn & CSS Armed chỉ khi trạng thái thay đổi
        armed = bool(data.get("armed", False))
        if armed != self._last_armed:
            self._last_armed = armed
            self.lbl_arm.setText("ARMED" if armed else "DISARMED")
            self.lbl_arm.setStyleSheet("color: #4ADE80; font-weight: bold; font-family: 'Consolas', monospace;" if armed else "color: #777777; font-family: 'Consolas', monospace;")

        # 4. Quản lý đồng hồ bay thực tế (Elapsed Flight Timer)
        if armed:
            if not self.prev_armed or self.arm_start_time is None:
                self.arm_start_time = time.time()
                self.prev_armed = True
            current_session = time.time() - self.arm_start_time
            total_sec = int(self.accumulated_flight_sec + current_session)
            mm, ss = divmod(total_sec, 60)
            self.lbl_elapsed.setText(f"{mm:02d}:{ss:02d}")
            elapsed_state = "armed"
        else:
            if self.prev_armed and self.arm_start_time is not None:
                self.accumulated_flight_sec += (time.time() - self.arm_start_time)
                self.arm_start_time = None
                self.prev_armed = False
            total_sec = int(self.accumulated_flight_sec)
            mm, ss = divmod(total_sec, 60)
            self.lbl_elapsed.setText(f"{mm:02d}:{ss:02d}" if total_sec > 0 else "00:00")
            elapsed_state = "paused" if total_sec > 0 else "zero"

        if elapsed_state != self._last_elapsed_state:
            self._last_elapsed_state = elapsed_state
            if elapsed_state == "armed":
                self.lbl_elapsed.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #4ADE80;")
            elif elapsed_state == "paused":
                self.lbl_elapsed.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #F59E0B;")
            else:
                self.lbl_elapsed.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #777777;")

        z = data.get("z", 0.0)
        spd = data.get("speed", 0.0)
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)

        # 5. Tính toán pin LiPo 3S và thời lượng còn lại
        bat_pct = calc_lipo_3s_pct(v)
        current_a = data.get("current_a", 0.0)
        if current_a >= 1.0:
            remaining_ah = (bat_pct / 100.0) * 1.4
            est_min = round((remaining_ah / current_a) * 60.0, 1)
        else:
            est_min = round((bat_pct / 100.0) * 12.1, 1)

        # 6. Đổi màu chỉ báo và thanh máu HP - Chỉ chạy khi đổi tầng phân nhóm
        bat_tier = 2 if bat_pct > 50 else (1 if bat_pct > 20 else 0)
        if bat_tier != self._last_bat_tier:
            self._last_bat_tier = bat_tier
            if bat_tier == 2:
                bat_color = "#4ADE80"
                grad_c1 = "#15803D"
                grad_c2 = "#22C55E"
                hp_tag_color = "#22C55E"
            elif bat_tier == 1:
                bat_color = "#FBBF24"
                grad_c1 = "#B45309"
                grad_c2 = "#F59E0B"
                hp_tag_color = "#F59E0B"
            else:
                bat_color = "#EF4444"
                grad_c1 = "#991B1B"
                grad_c2 = "#EF4444"
                hp_tag_color = "#EF4444"

            self.pbar_bat.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #141414;
                    border: 1px solid #222222;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {grad_c1}, stop:1 {grad_c2});
                    border-radius: 2px;
                }}
            """)
            self.lbl_hp_icon.setStyleSheet(f"font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; color: {hp_tag_color};")
            self.lbl_bat.setStyleSheet(f"font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; color: {bat_color};")

        self.pbar_bat.setValue(bat_pct)
        self.lbl_bat.setText(f"{v:.1f}V ({bat_pct}%)")

        # 7. Cập nhật các thông số lưới (cực nhẹ O(1))
        self.lbl_alt.setText(f"{z:.1f} m")
        self.lbl_spd.setText(f"{spd:.1f} m/s")
        self.lbl_time.setText(f"~{est_min:.1f}p")
        self.lbl_pos.setText(f"({x:.1f}m, {y:.1f}m)")
        self.lbl_curr.setText(f"{current_a:.1f} A")

        thr = data.get("throttle", 0)
        self.lbl_thr.setText(f"{thr}%")
        thr_tier = 2 if thr > 65 else (1 if thr > 20 else 0)
        if thr_tier != self._last_thr_tier:
            self._last_thr_tier = thr_tier
            if thr_tier == 2:
                self.lbl_thr.setStyleSheet("color: #F87171; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
            elif thr_tier == 1:
                self.lbl_thr.setStyleSheet("color: #A3E635; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")
            else:
                self.lbl_thr.setStyleSheet("color: #777777; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace;")

    def mouseDoubleClickEvent(self, event):
        # Click đúp vào thẻ khi đang DISARMED để đặt lại đồng hồ về 00:00 (bắt đầu chu kỳ bay mới)
        if not self.prev_armed:
            self.accumulated_flight_sec = 0.0
            self.lbl_elapsed.setText("00:00")
            self.lbl_elapsed.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; color: #777777;")
        super().mouseDoubleClickEvent(event)



class SwarmGCSWindow(QMainWindow):
    """Cửa sổ chính của Swarm Ground Control Station (Pure Black Tactical)."""
    def __init__(self, num_drones: int = DEFAULT_NUM_DRONES):
        super().__init__()
        self.num_drones = num_drones
        self.setWindowTitle("DRONE SWARM MISSION CONTROL — TACTICAL ARDUPILOT APF GCS")
        self.resize(1380, 860)
        self.setStyleSheet(MAIN_STYLESHEET)

        self.command_bridge = SwarmCommandBridge(project_root, parent=self)
        self.sim_manager = SwarmSimulationManager(project_root, parent=self)
        self.cards: Dict[int, DroneCard] = {}

        self._init_ui()
        self._start_telemetry_thread()

        # Kết nối signal an toàn giữa các QObject (QueuedConnection)
        self.sim_manager.status_updated.connect(self._on_sim_status_updated)
        self.sim_manager.log_emitted.connect(self._log)
        self.command_bridge.log_emitted.connect(self._log)
        self.command_bridge.mission_finished.connect(self._on_mission_finished)

    def _init_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        main_layout = QVBoxLayout(root_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Header Bar
        main_layout.addWidget(self._create_header())

        # 2. Main Body Splitter
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        body_splitter.setContentsMargins(10, 10, 10, 10)
        body_splitter.setHandleWidth(1)

        # Cột 1: Telemetry Deck (Bên trái)
        body_splitter.addWidget(self._create_telemetry_deck())

        # Cột 2: Tactical Radar Canvas (Ở giữa)
        self.radar_widget = TacticalRadarWidget()
        body_splitter.addWidget(self.radar_widget)

        # Cột 3: Mission & Command Deck (Bên phải)
        body_splitter.addWidget(self._create_command_deck())

        body_splitter.setCollapsible(0, False)
        body_splitter.setCollapsible(1, False)
        body_splitter.setCollapsible(2, False)

        body_splitter.setSizes([300, 680, 360])
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)
        body_splitter.setStretchFactor(2, 0)
        main_layout.addWidget(body_splitter, stretch=1)

        # 3. Bottom Console Log
        main_layout.addWidget(self._create_log_console())

    def _create_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 8, 14, 8)

        # App Title & Subtitle
        title_box = QVBoxLayout()
        title = QLabel("DRONE SWARM MISSION CONTROL")
        title.setObjectName("appTitle")
        sub = QLabel(f"Autonomous {self.num_drones}-Drone Formation Control | 3D Artificial Potential Field")
        sub.setStyleSheet("font-size: 10px; color: #666666; font-family: 'Consolas', monospace;")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        layout.addLayout(title_box)

        layout.addStretch()

        # Status Badges
        self.lbl_conn_status = QLabel(f"0/{self.num_drones} DRONES CONNECTED")
        self.lbl_conn_status.setObjectName("statusBadge")
        layout.addWidget(self.lbl_conn_status)

        self.lbl_min_dist = QLabel("MIN DIST: -- m")
        self.lbl_min_dist.setObjectName("statusBadge")
        layout.addWidget(self.lbl_min_dist)

        self.lbl_apf_status = QLabel("APF: NOMINAL")
        self.lbl_apf_status.setObjectName("statusBadgeNominal")
        layout.addWidget(self.lbl_apf_status)

        # Nút khẩn cấp
        btn_estop = QPushButton("🚨 STOP / LAND ALL")
        btn_estop.setObjectName("btnEmergency")
        btn_estop.clicked.connect(self._on_emergency_clicked)
        layout.addWidget(btn_estop)

        return frame

    def _create_telemetry_deck(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(6)

        header = QLabel(f"FLEET TELEMETRY ({self.num_drones} DRONES)")
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888888; font-family: 'Consolas', monospace; letter-spacing: 0.5px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(5)

        for i in range(1, self.num_drones + 1):
            card = DroneCard(i)
            self.cards[i] = card
            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def _create_command_deck(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        scroll.setMinimumWidth(320)

        panel = QWidget()
        panel.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 4, 0)
        layout.setSpacing(8)

        # 0. GroupBox: Simulation Process Manager (MÔ PHỎNG HỆ THỐNG)
        gb_sim = QGroupBox("MÔ PHỎNG HỆ THỐNG (SIMULATION SUITE)")
        sim_layout = QVBoxLayout(gb_sim)
        sim_layout.setSpacing(6)

        # Hàng nút Master: Bật tất cả & Tắt tất cả
        h_master = QHBoxLayout()
        btn_sim_all = QPushButton("⚡ BẬT TẤT CẢ (1-Click Sim)")
        btn_sim_all.setStyleSheet("background-color: #1A3022; border: 1px solid #2E7D32; color: #86EFAC; font-weight: bold; padding: 7px; border-radius: 3px; font-size: 11px;")
        btn_sim_all.clicked.connect(lambda: self.sim_manager.start_all_simulation(self.num_drones))

        btn_sim_stop = QPushButton("⏹ TẮT TOÀN BỘ")
        btn_sim_stop.setStyleSheet("background-color: #3B1111; border: 1px solid #991B1B; color: #FCA5A5; font-weight: bold; padding: 7px; border-radius: 3px; font-size: 11px;")
        btn_sim_stop.clicked.connect(self.sim_manager.stop_all_simulation)

        h_master.addWidget(btn_sim_all)
        h_master.addWidget(btn_sim_stop)
        sim_layout.addLayout(h_master)

        # Hàng nút đơn lẻ
        h_single = QHBoxLayout()
        btn_sitl = QPushButton(f"🚀 SITL {self.num_drones} Drone")
        btn_sitl.setStyleSheet("background-color: #141414; border: 1px solid #333333; color: #D4D4D4; font-size: 11px; padding: 5px; border-radius: 3px;")
        btn_sitl.clicked.connect(lambda: self.sim_manager.start_sitl(self.num_drones))

        btn_pub = QPushButton("📡 3D Mesh Pub")
        btn_pub.setStyleSheet("background-color: #141414; border: 1px solid #333333; color: #D4D4D4; font-size: 11px; padding: 5px; border-radius: 3px;")
        btn_pub.clicked.connect(lambda: self.sim_manager.start_mesh_publisher(self.num_drones))

        btn_rviz = QPushButton("👁️ Mở RViz2 3D")
        btn_rviz.setStyleSheet("background-color: #141414; border: 1px solid #333333; color: #D4D4D4; font-size: 11px; padding: 5px; border-radius: 3px;")
        btn_rviz.clicked.connect(lambda: self.sim_manager.start_rviz(self.num_drones))

        h_single.addWidget(btn_sitl)
        h_single.addWidget(btn_pub)
        h_single.addWidget(btn_rviz)
        sim_layout.addLayout(h_single)

        # Đèn LED trạng thái tiến trình
        h_led = QHBoxLayout()
        self.lbl_sitl_led = QLabel("SITL: [○]")
        self.lbl_sitl_led.setStyleSheet("color: #555555; font-size: 10px; font-family: 'Consolas', monospace;")
        self.lbl_pub_led = QLabel("PUB: [○]")
        self.lbl_pub_led.setStyleSheet("color: #555555; font-size: 10px; font-family: 'Consolas', monospace;")
        self.lbl_rviz_led = QLabel("RVIZ: [○]")
        self.lbl_rviz_led.setStyleSheet("color: #555555; font-size: 10px; font-family: 'Consolas', monospace;")

        h_led.addWidget(self.lbl_sitl_led)
        h_led.addWidget(self.lbl_pub_led)
        h_led.addWidget(self.lbl_rviz_led)
        sim_layout.addLayout(h_led)

        layout.addWidget(gb_sim)

        # 1. GroupBox: Global Flight Controls
        gb_flight = QGroupBox("ĐIỀU KHIỂN BAY ĐỒNG BỘ")
        flight_layout = QVBoxLayout(gb_flight)
        flight_layout.setSpacing(6)

        # Định nghĩa các stylesheet chuẩn xúc giác cho Command Deck
        self.style_arm_idle = "background-color: #064E3B; border: 1px solid #10B981; color: #A7F3D0; font-weight: bold; padding: 8px; border-radius: 4px; font-size: 11px;"
        self.style_arm_active = "background-color: #7F1D1D; border: 1px solid #EF4444; color: #FECACA; font-weight: bold; padding: 8px; border-radius: 4px; font-size: 11px;"
        self.style_takeoff = "background-color: #1E3A8A; border: 1px solid #3B82F6; color: #DBEAFE; font-weight: bold; padding: 8px; border-radius: 4px; font-size: 11px;"
        self.style_land = "background-color: #78350F; border: 1px solid #D97706; color: #FEF3C7; font-weight: bold; padding: 8px; border-radius: 4px; font-size: 11px;"

        self.style_form_idle = "background-color: #141414; border: 1px solid #333333; color: #D4D4D4; font-weight: 500; padding: 8px 6px; border-radius: 4px; font-size: 11px;"
        self.style_form_active = "background-color: #0C4A6E; border: 1px solid #38BDF8; color: #E0F2FE; font-weight: bold; padding: 8px 6px; border-radius: 4px; font-size: 11px;"

        self.style_mission = "background-color: #1E1B4B; border: 1px solid #4F46E5; color: #E0E7FF; font-weight: 600; padding: 9px 12px; border-radius: 4px; font-size: 11px; text-align: left;"
        self.style_perturb = "background-color: #431407; border: 1px solid #EA580C; color: #FFEDD5; font-weight: bold; padding: 9px 12px; border-radius: 4px; font-size: 11px; text-align: left;"

        # Thanh trượt độ cao cất cánh (Khóa an toàn phòng kín 1.0m - 2.0m, mặc định 1.3m)
        h_alt_box = QHBoxLayout()
        h_alt_box.addWidget(QLabel("Độ cao cất cánh:"))
        self.lbl_slider_val = QLabel("1.3m")
        self.lbl_slider_val.setStyleSheet("font-weight: bold; color: #EEEEEE; font-family: 'Consolas', monospace;")
        h_alt_box.addStretch()
        h_alt_box.addWidget(self.lbl_slider_val)
        flight_layout.addLayout(h_alt_box)

        self.slider_alt = QSlider(Qt.Orientation.Horizontal)
        self.slider_alt.setRange(10, 20)  # 1.0m đến 2.0m (giữ khoảng đệm an toàn dưới trần 2.5m)
        self.slider_alt.setValue(13)       # Mặc định 1.3m
        self.slider_alt.valueChanged.connect(lambda val: self.lbl_slider_val.setText(f"{val/10:.1f}m"))
        flight_layout.addWidget(self.slider_alt)

        # Hàng nút: ARM & TAKEOFF
        h_btn_box = QHBoxLayout()
        self.btn_arm = QPushButton("⚔️ ARM TOÀN BỘ")
        self.btn_arm.setStyleSheet(self.style_arm_idle)
        self.btn_arm.clicked.connect(self._on_arm_clicked)

        self.btn_takeoff = QPushButton("🛫 CẤT CÁNH")
        self.btn_takeoff.setStyleSheet(self.style_takeoff)
        self.btn_takeoff.clicked.connect(self._on_takeoff_clicked)

        h_btn_box.addWidget(self.btn_arm)
        h_btn_box.addWidget(self.btn_takeoff)
        flight_layout.addLayout(h_btn_box)

        # Nút LAND
        self.btn_land = QPushButton("🛬 HẠ CÁNH ĐỒNG BỘ (LAND)")
        self.btn_land.setStyleSheet(self.style_land)
        self.btn_land.clicked.connect(self._on_land_clicked)
        flight_layout.addWidget(self.btn_land)

        layout.addWidget(gb_flight)

        # 2. GroupBox: Safety & Battery Failsafe Suite (Đặt ngay dưới cụm bay để truy cập tức thời)
        gb_safety = QGroupBox("BẢO VỆ AN TOÀN & TEST FAILSAFE (SIM)")
        safety_layout = QVBoxLayout(gb_safety)
        safety_layout.setSpacing(6)

        h_safety_btn = QHBoxLayout()
        self.btn_fs_inject = QPushButton("⚡ Sụt Áp D1 (10.2V)")
        self.btn_fs_inject.setStyleSheet("background-color: #450A0A; border: 1px solid #DC2626; color: #FCA5A5; font-weight: bold; padding: 7px; border-radius: 4px; font-size: 11px;")
        self.btn_fs_inject.clicked.connect(self._on_inject_failsafe_clicked)

        self.btn_fs_reset = QPushButton("🔋 Nạp Pin Toàn Bầy (12.6V)")
        self.btn_fs_reset.setStyleSheet("background-color: #064E3B; border: 1px solid #059669; color: #6EE7B7; font-weight: bold; padding: 7px; border-radius: 4px; font-size: 11px;")
        self.btn_fs_reset.clicked.connect(self._on_reset_battery_clicked)

        h_safety_btn.addWidget(self.btn_fs_inject)
        h_safety_btn.addWidget(self.btn_fs_reset)
        safety_layout.addLayout(h_safety_btn)

        layout.addWidget(gb_safety)

        # 3. GroupBox: Formation Morphing
        gb_formation = QGroupBox("BIẾN ĐỔI ĐỘI HÌNH (APF MORPHING)")
        form_layout = QGridLayout(gb_formation)
        form_layout.setSpacing(6)

        self.formation_buttons = {}

        btn_v = QPushButton("✦ Chữ V (V-Shape)")
        btn_v.setStyleSheet(self.style_form_idle)
        btn_v.clicked.connect(lambda: self._on_formation_clicked("V_SHAPE"))
        self.formation_buttons["V_SHAPE"] = btn_v

        btn_dia = QPushButton("◆ Kim Cương (Diamond)")
        btn_dia.setStyleSheet(self.style_form_idle)
        btn_dia.clicked.connect(lambda: self._on_formation_clicked("DIAMOND"))
        self.formation_buttons["DIAMOND"] = btn_dia

        btn_col = QPushButton("❙ Hàng Dọc (Column)")
        btn_col.setStyleSheet(self.style_form_active)  # Mặc định xuất phát
        btn_col.clicked.connect(lambda: self._on_formation_clicked("COLUMN"))
        self.formation_buttons["COLUMN"] = btn_col

        btn_line = QPushButton("━ Hàng Ngang (Line)")
        btn_line.setStyleSheet(self.style_form_idle)
        btn_line.clicked.connect(lambda: self._on_formation_clicked("LINE"))
        self.formation_buttons["LINE"] = btn_line

        form_layout.addWidget(btn_v, 0, 0)
        form_layout.addWidget(btn_dia, 0, 1)
        form_layout.addWidget(btn_col, 1, 0)
        form_layout.addWidget(btn_line, 1, 1)

        layout.addWidget(gb_formation)

        # 3. GroupBox: Autonomous Mission Runner
        gb_mission = QGroupBox("KỊCH BẢN BAY TỰ HÀNH (ORCA & APF)")
        mission_layout = QVBoxLayout(gb_mission)
        mission_layout.setSpacing(6)

        btn_m_orca_snap = QPushButton("🛡️ Show 30 Năm (ORCA - Minimum Snap)")
        btn_m_orca_snap.setStyleSheet("background-color: #064E3B; border: 1px solid #10B981; color: #D1FAE5; font-weight: bold; padding: 9px 12px; border-radius: 4px; font-size: 11px; text-align: left;")
        btn_m_orca_snap.clicked.connect(lambda: self._on_launch_mission("orca_swarm/show_30years_orca.py", extra_cli=["--traj", "min_snap"]))

        btn_m_orca_spline = QPushButton("🌀 Show 30 Năm (ORCA - B-Spline)")
        btn_m_orca_spline.setStyleSheet("background-color: #064E3B; border: 1px solid #10B981; color: #D1FAE5; font-weight: bold; padding: 9px 12px; border-radius: 4px; font-size: 11px; text-align: left;")
        btn_m_orca_spline.clicked.connect(lambda: self._on_launch_mission("orca_swarm/show_30years_orca.py", extra_cli=["--traj", "bspline"]))

        btn_m1 = QPushButton("🎓 Show 30 Năm (APF Smooth)")
        btn_m1.setStyleSheet(self.style_mission)
        btn_m1.clicked.connect(lambda: self._on_launch_mission("missions/show_30years_smooth.py"))

        btn_m2 = QPushButton("🔄 Bầy Đàn APF Leader-Follower")
        btn_m2.setStyleSheet(self.style_mission)
        btn_m2.clicked.connect(lambda: self._on_launch_mission("missions/swarm_apf_leader_follower.py"))

        btn_perturb = QPushButton("⚡ Test Xung Nhiễu Né Tránh APF")
        btn_perturb.setStyleSheet(self.style_perturb)
        btn_perturb.clicked.connect(self._on_perturbation_clicked)

        btn_swap = QPushButton("🔀 Đổi Chỗ Đối Đầu (Chống Đi Xuyên)")
        btn_swap.setStyleSheet("background-color: #2E1065; border: 1px solid #9333EA; color: #F3E8FF; font-weight: bold; padding: 9px 12px; border-radius: 4px; font-size: 11px; text-align: left;")
        btn_swap.clicked.connect(self._on_swap_clicked)

        mission_layout.addWidget(btn_m_orca_snap)
        mission_layout.addWidget(btn_m_orca_spline)
        mission_layout.addWidget(btn_m1)
        mission_layout.addWidget(btn_m2)
        mission_layout.addWidget(btn_perturb)
        mission_layout.addWidget(btn_swap)

        layout.addWidget(gb_mission)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _create_log_console(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet("background-color: #000000; border-top: 1px solid #1A1A1A;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(3)

        header_box = QHBoxLayout()
        lbl_log = QLabel("EVENT & MISSION LOG STREAM")
        lbl_log.setStyleSheet("font-size: 10px; font-weight: bold; color: #666666; font-family: 'Consolas', monospace;")
        btn_clear = QPushButton("Xóa Log")
        btn_clear.setMaximumWidth(60)
        btn_clear.setStyleSheet("font-size: 10px; padding: 1px 4px; background-color: #141414; border: 1px solid #222222; color: #888888;")
        btn_clear.clicked.connect(lambda: self.txt_log.clear())

        header_box.addWidget(lbl_log)
        header_box.addStretch()
        header_box.addWidget(btn_clear)
        layout.addLayout(header_box)

        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("logConsole")
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(110)
        self.txt_log.document().setMaximumBlockCount(1000)
        layout.addWidget(self.txt_log)

        return frame

    def _start_telemetry_thread(self):
        """Khởi động luồng đọc Telemetry nền."""
        self.worker = SwarmTelemetryWorker(num_drones=self.num_drones)
        self.worker.telemetry_updated.connect(self._on_telemetry_updated)
        self.worker.log_message.connect(self._log)
        self.worker.start()

    def _on_sim_status_updated(self, sitl_on: bool, pub_on: bool, rviz_on: bool):
        """Cập nhật đèn LED trạng thái tiến trình mô phỏng."""
        self.lbl_sitl_led.setText("SITL: [●]" if sitl_on else "SITL: [○]")
        self.lbl_sitl_led.setStyleSheet("color: #4ADE80; font-weight: bold; font-family: 'Consolas', monospace;" if sitl_on else "color: #555555; font-family: 'Consolas', monospace;")

        self.lbl_pub_led.setText("PUB: [●]" if pub_on else "PUB: [○]")
        self.lbl_pub_led.setStyleSheet("color: #4ADE80; font-weight: bold; font-family: 'Consolas', monospace;" if pub_on else "color: #555555; font-family: 'Consolas', monospace;")

        self.lbl_rviz_led.setText("RVIZ: [●]" if rviz_on else "RVIZ: [○]")
        self.lbl_rviz_led.setStyleSheet("color: #4ADE80; font-weight: bold; font-family: 'Consolas', monospace;" if rviz_on else "color: #555555; font-family: 'Consolas', monospace;")

    def _on_telemetry_updated(self, payload: dict):
        """Callback cập nhật GUI khi có dữ liệu telemetry mới (20Hz)."""
        # 1. Cập nhật Radar ngay tức khắc để giữ luồng nội suy 60 FPS mượt mà
        self.radar_widget.update_telemetry(payload)

        # 2. Phân tách tần số: Cập nhật thẻ thông số và Header ở 10Hz (tiết kiệm 65% CPU cho luồng UI chính)
        now = time.monotonic()
        if hasattr(self, "_last_heavy_ui_t") and (now - self._last_heavy_ui_t) < 0.09:
            return
        self._last_heavy_ui_t = now

        drones = payload.get("drones", {})
        conn_count = payload.get("connected_count", 0)
        min_dist = payload.get("min_distance_m", 999.0)
        safe_status = payload.get("safe_status", "NOMINAL")

        # Cập nhật thẻ từng drone
        for d_id, d_data in drones.items():
            if d_id in self.cards:
                self.cards[d_id].update_data(d_data)

        # Cập nhật Header
        self.lbl_conn_status.setText(f"{conn_count}/{self.num_drones} DRONES CONNECTED")
        target_conn_badge = "statusBadgeNominal" if conn_count == self.num_drones else ("statusBadgeWarning" if conn_count > 0 else "statusBadge")
        if self.lbl_conn_status.objectName() != target_conn_badge:
            self.lbl_conn_status.setObjectName(target_conn_badge)
            self.lbl_conn_status.style().unpolish(self.lbl_conn_status)
            self.lbl_conn_status.style().polish(self.lbl_conn_status)

        self.lbl_min_dist.setText(f"MIN DIST: {min_dist:.2f}m")

        # Kiểm tra trạng thái Failsafe trên toàn bầy
        fs_drones = [str(d_id) for d_id, d in drones.items() if d.get("failsafe_active") or (d.get("voltage", 12.6) <= 10.5 and d.get("mode") == "LAND")]
        if fs_drones:
            self.lbl_apf_status.setText(f"🚨 FAILSAFE: D{','.join(fs_drones)} LOW BATT")
            target_apf_badge = "statusBadgeCritical"
        else:
            self.lbl_apf_status.setText(f"APF: {safe_status}")
            if safe_status == "CRITICAL_AVOID":
                target_apf_badge = "statusBadgeCritical"
            elif safe_status in ("APF_ACTIVE", "WARNING_CLOSE"):
                target_apf_badge = "statusBadgeWarning"
            else:
                target_apf_badge = "statusBadgeNominal"

        if self.lbl_apf_status.objectName() != target_apf_badge:
            self.lbl_apf_status.setObjectName(target_apf_badge)
            self.lbl_apf_status.style().unpolish(self.lbl_apf_status)
            self.lbl_apf_status.style().polish(self.lbl_apf_status)

        # Đồng bộ hóa trạng thái nút ARM theo thực tế telemetry
        is_any_armed = any(d.get("armed", False) for d in drones.values())
        cur_text = self.btn_arm.text()
        if is_any_armed:
            if cur_text != "🛑 DISARM TOÀN BỘ":
                self.btn_arm.setText("🛑 DISARM TOÀN BỘ")
                self.btn_arm.setStyleSheet(self.style_arm_active)
        else:
            if cur_text not in ("⚔️ ARM TOÀN BỘ", "⏳ ĐANG ARM..."):
                self.btn_arm.setText("⚔️ ARM TOÀN BỘ")
                self.btn_arm.setStyleSheet(self.style_arm_idle)

    def _sync_arm_button_state(self):
        """Khôi phục trạng thái hiển thị nút ARM sau thời gian chờ kết quả."""
        if not hasattr(self, "worker"):
            return
        is_any_armed = any(d.get("armed", False) for d in self.worker.raw_data.values())
        if is_any_armed:
            self.btn_arm.setText("🛑 DISARM TOÀN BỘ")
            self.btn_arm.setStyleSheet(self.style_arm_active)
        else:
            self.btn_arm.setText("⚔️ ARM TOÀN BỘ")
            self.btn_arm.setStyleSheet(self.style_arm_idle)

    def _log(self, level: str, msg: str):
        """Ghi log vào console với phong cách kỹ thuật thuần túy (Thread-safe & HTML escaped)."""
        import html
        now_str = time.strftime("%H:%M:%S")
        color_map = {
            "INFO": "#94A3B8",
            "SUCCESS": "#4ADE80",
            "WARNING": "#FBBF24",
            "ERROR": "#F87171",
            "CRITICAL": "#EF4444",
            "MISSION": "#A78BFA",
        }
        color = color_map.get(level, "#888888")
        escaped_msg = html.escape(str(msg))
        html_line = f"<span style='color:#555555;'>[{now_str}]</span> <b style='color:{color};'>[{level}]</b> <span style='color:#CCCCCC;'>{escaped_msg}</span>"
        self.txt_log.append(html_line)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # --- ACTIONS & COMMANDS ---
    def _on_arm_clicked(self):
        is_any_armed = any(d.get("armed", False) for d in self.worker.raw_data.values()) if hasattr(self, "worker") else False

        if not is_any_armed:
            self._log("INFO", f"Phát lệnh ARM (GUIDED) trực tiếp cho cả {self.num_drones} Drone...")
            self.btn_arm.setText("⏳ ĐANG ARM...")
            self.btn_arm.setStyleSheet(self.style_arm_active)
            if hasattr(self, "worker"):
                self.worker.arm_all(True)
            # Sau 4.5s tự động đối soát lại telemetry thực tế để cập nhật nút
            QTimer.singleShot(4500, self._sync_arm_button_state)
        else:
            self._log("WARNING", f"Phát lệnh DISARM đồng loạt cho {self.num_drones} Drone...")
            if hasattr(self, "worker"):
                self.worker.arm_all(False)
            self.btn_arm.setText("⚔️ ARM TOÀN BỘ")
            self.btn_arm.setStyleSheet(self.style_arm_idle)

    def _on_takeoff_clicked(self):
        target_alt = self.slider_alt.value() / 10.0
        self._log("INFO", f"Phát lệnh cất cánh GUIDED Takeoff đồng bộ lên độ cao: {target_alt:.1f}m")
        if hasattr(self, "worker"):
            self.worker.takeoff_all(target_alt)
        self.btn_arm.setText("🛑 DISARM TOÀN BỘ")
        self.btn_arm.setStyleSheet(self.style_arm_active)

    def _on_land_clicked(self):
        self._log("WARNING", "Yêu cầu toàn bộ bầy Drone hạ cánh khẩn cấp (LAND)...")
        if hasattr(self, "worker"):
            self.worker.land_all()
        self.command_bridge.emergency_stop(on_log_callback=self._log)
        self.btn_arm.setText("⚔️ ARM TOÀN BỘ")
        self.btn_arm.setStyleSheet(self.style_arm_idle)

    def _on_emergency_clicked(self):
        self._log("CRITICAL", "🚨 ĐÃ BẤM NÚT NGẮT KHẨN CẤP! Dừng toàn bộ kịch bản và LAND lập tức!")
        if hasattr(self, "worker"):
            self.worker.land_all()
        self.command_bridge.emergency_stop(on_log_callback=self._log)
        self.btn_arm.setText("⚔️ ARM TOÀN BỘ")
        self.btn_arm.setStyleSheet(self.style_arm_idle)

    def _on_formation_clicked(self, formation_type: str):
        target_alt = self.slider_alt.value() / 10.0
        for f_type, btn in self.formation_buttons.items():
            if f_type == formation_type:
                btn.setStyleSheet(self.style_form_active)
            else:
                btn.setStyleSheet(self.style_form_idle)

        if hasattr(self, "worker"):
            self.worker.set_formation(formation_type, altitude=target_alt)

    def _on_launch_mission(self, script_path: str, extra_cli: list = None):
        try:
            extra_args = []
            if hasattr(self, "sim_manager") and self.sim_manager.is_sitl_running():
                extra_args.append("--sitl")
            if extra_cli:
                extra_args.extend(extra_cli)
            self._log("INFO", f"Khởi chạy kịch bản bay tự hành: {script_path} (tham số: {extra_args if extra_args else 'chuẩn'})")
            success = self.command_bridge.launch_script_async(script_path, extra_args=extra_args)
            if success:
                self.btn_arm.setText("🛑 DISARM TOÀN BỘ")
                self.btn_arm.setStyleSheet(self.style_arm_active)
        except Exception as e:
            self._log("ERROR", f"Lỗi khởi động kịch bản: {e}")

    def _on_mission_finished(self, return_code: int):
        """Callback nhận kết quả khi kịch bản kết thúc."""
        if return_code == 0:
            self._log("SUCCESS", "✅ Kịch bản bay đã hoàn thành an toàn!")
        else:
            self._log("WARNING", f"⚠️ Kịch bản bay kết thúc với mã thoát: {return_code}")
        self._sync_arm_button_state()

    def _on_perturbation_clicked(self):
        target_alt = self.slider_alt.value() / 10.0
        self._log("WARNING", "⚡ Kích hoạt xung nhiễu cưỡng bức: Ép Drone 2 tiến sát Drone 3 để kiểm thử phản xạ APF né tránh!")
        if hasattr(self, "worker"):
            self.worker.send_perturbation_stream(drone_id=2, target_id=3, altitude=target_alt)

    def _on_swap_clicked(self):
        target_alt = self.slider_alt.value() / 10.0
        self._log("WARNING", "🔀 Kích hoạt kịch bản ĐỔI CHỖ ĐỐI ĐẦU (Head-on Swap): Drone 2 ⇄ Drone 3 (Quan sát 2 drone tự lượn chữ S né nhau mà không đi xuyên qua nhau)!")
        if hasattr(self, "worker"):
            self.worker.run_head_on_swap_demo(altitude=target_alt)

    def _on_inject_failsafe_clicked(self):
        self._log("CRITICAL", "⚡ ĐÃ BẮN LỆNH SỰ CỐ SỤT ÁP: Bơm tham số SIM_BATT_VOLTAGE = 10.2V vào Drone 1 (ngưỡng Failsafe 10.5V)...")
        if hasattr(self, "worker"):
            ok = self.worker.inject_sim_battery_voltage(1, 10.2)
            if ok:
                self._log("WARNING", "Chờ 3 giây bộ lọc BATT_LOW_TIMER để quan sát ArduPilot tự ép chuyển mode LAND!")
            else:
                self._log("ERROR", "Không thể gửi tham số tới Drone 1! Hãy chắc chắn Drone 1 đã kết nối MAVLink.")

    def _on_reset_battery_clicked(self):
        self._log("SUCCESS", f"🔋 ĐÃ PHỤC HỒI NGUỒN PIN: Đặt lại SIM_BATT_VOLTAGE = 12.6V cho toàn bộ {self.num_drones} Drone (Pin 3S đầy 100%)...")
        if hasattr(self, "worker"):
            for d in range(1, self.num_drones + 1):
                self.worker.inject_sim_battery_voltage(d, 12.6)

    def closeEvent(self, event):
        """Dọn dẹp sạch sẽ luồng nền khi đóng cửa sổ."""
        self._log("INFO", "Đang đóng ứng dụng GCS...")
        if hasattr(self, "worker"):
            self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    num_drones = DEFAULT_NUM_DRONES
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        num_drones = int(sys.argv[1])
    window = SwarmGCSWindow(num_drones=num_drones)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
