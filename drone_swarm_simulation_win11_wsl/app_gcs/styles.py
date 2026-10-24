#!/usr/bin/env python3
"""
DRONE SWARM GCS - STYLESHEET & DESIGN SYSTEM
Theme: Pure Black Tactical Engineering (Tối giản, chuyên nghiệp, không màu mè AI)
"""

# Bảng màu chuẩn nhận diện 5 Drone
DRONE_COLORS = {
    1: {"hex": "#E53935", "name": "Đỏ", "rgb": (229, 57, 53)},       # Leader
    2: {"hex": "#43A047", "name": "Xanh lá", "rgb": (67, 160, 71)},
    3: {"hex": "#1E88E5", "name": "Xanh lam", "rgb": (30, 136, 229)},
    4: {"hex": "#FB8C00", "name": "Vàng cam", "rgb": (251, 140, 0)},
    5: {"hex": "#8E24AA", "name": "Tím", "rgb": (142, 36, 170)},
}

# Cấu hình số lượng drone mặc định (3 Drone)
DEFAULT_NUM_DRONES = 3

# Tọa độ bãi đỗ mặt đất chuẩn hội trường 20m x 30m (bước 1.6m, span 6.4m, nằm trọn trong FENCE_RADIUS 12.0m)
SWARM_GROUND_OFFSETS = {
    1: (0.0, 0.0),
    2: (0.0, 1.6),
    3: (0.0, 3.2),
    4: (0.0, 4.8),
    5: (0.0, 6.4),
}

MAIN_STYLESHEET = """
/* --- PURE BLACK ENGINEERING THEME --- */
QMainWindow {
    background-color: #000000;
}

QWidget {
    font-family: 'Consolas', 'Segoe UI', monospace, sans-serif;
    color: #CCCCCC;
    font-size: 12px;
    background-color: transparent;
}

/* --- HEADER BAR --- */
QFrame#headerFrame {
    background-color: #0A0A0A;
    border-bottom: 1px solid #222222;
    padding: 6px 12px;
}

QLabel#appTitle {
    font-size: 15px;
    font-weight: bold;
    color: #FFFFFF;
    letter-spacing: 0.5px;
}

QLabel#statusBadge {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 2px;
    background-color: #161616;
    color: #AAAAAA;
    border: 1px solid #333333;
}

QLabel#statusBadgeNominal {
    background-color: #0D2818;
    color: #4ADE80;
    border: 1px solid #166534;
    padding: 3px 8px;
    border-radius: 2px;
    font-weight: bold;
}

QLabel#statusBadgeWarning {
    background-color: #2D1A04;
    color: #FBBF24;
    border: 1px solid #854D0E;
    padding: 3px 8px;
    border-radius: 2px;
    font-weight: bold;
}

QLabel#statusBadgeCritical {
    background-color: #3B0D0C;
    color: #F87171;
    border: 1px solid #991B1B;
    padding: 3px 8px;
    border-radius: 2px;
    font-weight: bold;
}

/* --- GROUP BOXES --- */
QGroupBox {
    background-color: #0A0A0A;
    border: 1px solid #222222;
    border-radius: 4px;
    margin-top: 20px;
    padding-top: 12px;
    font-weight: bold;
    font-size: 11px;
    color: #888888;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    top: 4px;
    padding: 1px 6px;
    background-color: #141414;
    border: 1px solid #2A2A2A;
    border-radius: 2px;
    color: #CCCCCC;
}

/* --- DRONE CARDS --- */
QFrame.droneCard {
    background-color: #0D0D0D;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 4px;
}

QFrame.droneCard:hover {
    border-color: #333333;
    background-color: #121212;
}

QLabel.cardHeader {
    font-size: 12px;
    font-weight: bold;
}

QLabel.metricValue {
    font-family: 'Consolas', monospace;
    font-size: 12px;
    font-weight: bold;
    color: #EEEEEE;
}

QLabel.metricLabel {
    font-size: 10px;
    color: #666666;
}

/* --- BUTTONS --- */
QPushButton {
    background-color: #1A1A1A;
    border: 1px solid #333333;
    border-radius: 3px;
    color: #E2E8F0;
    font-weight: 600;
    font-family: 'Consolas', monospace;
    padding: 7px 14px;
    min-height: 24px;
}

QPushButton:hover {
    background-color: #262626;
    border: 1px solid #4B5563;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: #111111;
    border: 1px solid #222222;
}

QPushButton#btnSimAction {
    background-color: #1E3A5F;
    border: 1px solid #3B82F6;
    color: #93C5FD;
    font-weight: bold;
}
QPushButton#btnSimAction:hover {
    background-color: #2563EB;
    border: 1px solid #60A5FA;
    color: #FFFFFF;
}

QPushButton#btnArm {
    background-color: #064E3B;
    border: 1px solid #10B981;
    color: #A7F3D0;
    font-weight: bold;
    min-height: 28px;
}
QPushButton#btnArm:hover {
    background-color: #047857;
    border: 1px solid #34D399;
    color: #FFFFFF;
}
QPushButton#btnArm[armed="true"] {
    background-color: #7F1D1D;
    border: 1px solid #EF4444;
    color: #FECACA;
}
QPushButton#btnArm[armed="true"]:hover {
    background-color: #991B1B;
    border: 1px solid #F87171;
}

QPushButton#btnTakeoff {
    background-color: #1E40AF;
    border: 1px solid #3B82F6;
    color: #DBEAFE;
    font-weight: bold;
    min-height: 28px;
}
QPushButton#btnTakeoff:hover {
    background-color: #2563EB;
    border: 1px solid #60A5FA;
    color: #FFFFFF;
}

QPushButton#btnLand {
    background-color: #78350F;
    border: 1px solid #D97706;
    color: #FEF3C7;
    font-weight: bold;
    min-height: 28px;
}
QPushButton#btnLand:hover {
    background-color: #92400E;
    border: 1px solid #F59E0B;
    color: #FFFFFF;
}

QPushButton#btnEmergency {
    background-color: #7F1D1D;
    border: 1px solid #DC2626;
    color: #FEE2E2;
    font-weight: bold;
    padding: 6px 14px;
}
QPushButton#btnEmergency:hover {
    background-color: #991B1B;
    border: 1px solid #EF4444;
    color: #FFFFFF;
}

QPushButton#btnFormation {
    background-color: #141414;
    border: 1px solid #383838;
    color: #E2E8F0;
    font-weight: 500;
    min-height: 26px;
    padding: 6px 8px;
}
QPushButton#btnFormation:hover {
    background-color: #1E293B;
    border: 1px solid #38BDF8;
    color: #38BDF8;
}
QPushButton#btnFormation[active="true"] {
    background-color: #0C2847;
    border: 1px solid #0EA5E9;
    color: #38BDF8;
    font-weight: bold;
}

QPushButton#btnMission {
    background-color: #1E1B4B;
    border: 1px solid #4F46E5;
    color: #E0E7FF;
    font-weight: 500;
    text-align: left;
    padding: 8px 12px;
    min-height: 28px;
}
QPushButton#btnMission:hover {
    background-color: #312E81;
    border: 1px solid #6366F1;
    color: #FFFFFF;
}

/* --- SLIDERS --- */
QSlider::groove:horizontal {
    height: 4px;
    background: #222222;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #CCCCCC;
    border: 1px solid #555555;
    width: 12px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background: #FFFFFF;
    border-color: #3B82F6;
}

/* --- LOG CONSOLE --- */
QTextEdit#logConsole {
    background-color: #050505;
    border: 1px solid #1A1A1A;
    border-radius: 2px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    color: #888888;
    padding: 4px;
}

/* --- SCROLLBAR --- */
QScrollBar:vertical {
    background: #000000;
    width: 6px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #222222;
    min-height: 20px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #333333;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
