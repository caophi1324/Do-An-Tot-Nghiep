#!/usr/bin/env python3
"""
DRONE SWARM GCS - TACTICAL 2D RADAR CANVAS (PURE BLACK THEME)
Hiển thị không gian bầy đàn 5 Drone, lưới toạ độ mét, liên kết Topo và cảnh báo APF.
"""

import math
from collections import deque
from typing import Dict, List, Tuple
from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget

from .styles import DRONE_COLORS


class TacticalRadarWidget(QWidget):
    """
    Widget radar hiển thị toàn cảnh vị trí 5 Drone trong không gian phẳng (X, Y)
    Giao diện tối giản quân sự nền đen tuyền (Pure Black).
    Tích hợp bộ nội suy thị giác 60 FPS (Visual Lerp Smoothing) siêu mượt như Gazebo.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(450, 450)
        self.setStyleSheet("background-color: #000000; border: 1px solid #1A1A1A; border-radius: 2px;")

        self.drones_data: Dict[int, dict] = {}
        self.trail_history: Dict[int, deque] = {}
        self.warning_pairs: List[Tuple[int, int, float]] = []

        # Tọa độ nội suy hiển thị 60 FPS (Visual Lerp States)
        self.render_poses: Dict[int, dict] = {}

        # Cấu hình khung nhìn radar (tâm tại X=2.0m, Y=3.2m bao trọn hội trường 20x30m & FENCE_RADIUS 12.0m)
        self.center_world_x = 2.0
        self.center_world_y = 3.2
        self.scale_pixels_per_meter = 25.0
        self.zoom_factor = 1.0

        self.dragging = False
        self.last_mouse_pos = None

        # Timer đồ họa 60 FPS độc lập (16.6ms) - Triệt tiêu 100% giật khựng trên màn hình App
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_frame)
        self.anim_timer.start()

    def update_telemetry(self, telemetry_payload: dict):
        """Tiếp nhận dữ liệu telemetry 20Hz từ Worker và nạp vào toạ độ đích."""
        self.drones_data = telemetry_payload.get("drones", {})
        self.warning_pairs = telemetry_payload.get("warning_pairs", [])

    def _on_anim_frame(self):
        """Vòng lặp đồ họa 60 FPS: Nội suy tuyến tính trơn mượt (Exponential Moving Average / Lerp)."""
        if not self.drones_data:
            return

        alpha = 0.32  # Hệ số làm mịn tối ưu: hấp thụ 100% micro-jitter mà không bị trễ
        needs_repaint = False

        for d_id, target in self.drones_data.items():
            tx = float(target.get("x", 0.0))
            ty = float(target.get("y", 0.0))
            tz = float(target.get("z", 0.0))
            tyaw = float(target.get("yaw", 0.0))
            connected = bool(target.get("connected", False))

            if d_id not in self.render_poses:
                self.render_poses[d_id] = {
                    "x": tx, "y": ty, "z": tz, "yaw": tyaw, "connected": connected
                }
                needs_repaint = True
            else:
                rp = self.render_poses[d_id]
                # Nội suy vị trí 3D
                rp["x"] += alpha * (tx - rp["x"])
                rp["y"] += alpha * (ty - rp["y"])
                rp["z"] += alpha * (tz - rp["z"])
                # Nội suy góc Yaw theo cung tròn ngắn nhất
                dyaw = (tyaw - rp["yaw"] + 180.0) % 360.0 - 180.0
                rp["yaw"] += alpha * dyaw
                rp["connected"] = connected
                needs_repaint = True

                # Cập nhật vệt đường bay mượt mà khi di chuyển > 3cm
                if connected:
                    if d_id not in self.trail_history:
                        self.trail_history[d_id] = deque(maxlen=30)
                    trail = self.trail_history[d_id]
                    if not trail or math.hypot(rp["x"] - trail[-1][0], rp["y"] - trail[-1][1]) > 0.03:
                        trail.append((rp["x"], rp["y"]))

        if needs_repaint or self.dragging:
            self.update()

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        scale = self.scale_pixels_per_meter * self.zoom_factor

        sx = cx + (wx - self.center_world_x) * scale
        sy = cy - (wy - self.center_world_y) * scale
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        scale = self.scale_pixels_per_meter * self.zoom_factor

        wx = self.center_world_x + (sx - cx) / scale
        wy = self.center_world_y - (sy - cy) / scale
        return wx, wy

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor = min(2.5, self.zoom_factor * 1.1)
        else:
            self.zoom_factor = max(0.4, self.zoom_factor * 0.9)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos:
            diff = event.pos() - self.last_mouse_pos
            scale = self.scale_pixels_per_meter * self.zoom_factor
            self.center_world_x -= diff.x() / scale
            self.center_world_y += diff.y() / scale
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Vẽ Lưới Radar tối giản
        self._draw_grid(painter)

        # 2. Vẽ Đường nối Topo Bầy đàn (Màu xanh mảnh)
        self._draw_topology_links(painter)

        # 3. Vẽ Vùng Cảnh báo Lực đẩy APF (Đỏ)
        self._draw_apf_warnings(painter)

        # 4. Vẽ Vệt đường bay
        self._draw_trails(painter)

        # 5. Vẽ 5 Drone
        self._draw_drones(painter)

        # 6. Vẽ Chú thích
        self._draw_overlay(painter)

    def _draw_grid(self, p: QPainter):
        w = self.width()
        h = self.height()
        scale = self.scale_pixels_per_meter * self.zoom_factor

        # Lưới ô vuông 1m
        p.setPen(QPen(QColor(22, 22, 22), 1, Qt.PenStyle.DotLine))
        min_wx, max_wy = self.screen_to_world(0, 0)
        max_wx, min_wy = self.screen_to_world(w, h)

        start_x = math.floor(min_wx)
        end_x = math.ceil(max_wx)
        for x in range(start_x, end_x + 1):
            sx, _ = self.world_to_screen(x, 0)
            p.drawLine(int(sx), 0, int(sx), h)

        start_y = math.floor(min_wy)
        end_y = math.ceil(max_wy)
        for y in range(start_y, end_y + 1):
            _, sy = self.world_to_screen(0, y)
            p.drawLine(0, int(sy), w, int(sy))

        # Trục toạ độ gốc (0, 0)
        orig_sx, orig_sy = self.world_to_screen(0.0, 0.0)
        p.setPen(QPen(QColor(50, 50, 50), 1))
        p.drawLine(0, int(orig_sy), w, int(orig_sy))
        p.drawLine(int(orig_sx), 0, int(orig_sx), h)

        # Vòng tròn cự ly (2m, 4m, 6m, 8m, 10m, 12m - chuẩn hội trường 20x30m)
        cx, cy = self.world_to_screen(self.center_world_x, self.center_world_y)
        p.setPen(QPen(QColor(32, 32, 32), 1, Qt.PenStyle.DashLine))
        for r_meter in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
            r_px = r_meter * scale
            p.drawEllipse(QPointF(cx, cy), r_px, r_px)

    def _draw_topology_links(self, p: QPainter):
        """Vẽ đường liên kết topo mỏng màu lam ngọc theo toạ độ nội suy 60 FPS."""
        d1 = self.render_poses.get(1) or self.drones_data.get(1)
        if not d1 or not d1.get("connected", False):
            return

        sx1, sy1 = self.world_to_screen(d1["x"], d1["y"])
        p.setPen(QPen(QColor(0, 180, 216, 120), 1, Qt.PenStyle.SolidLine))

        for f_id in range(2, 6):
            df = self.render_poses.get(f_id) or self.drones_data.get(f_id)
            if df and df.get("connected", False):
                sxf, syf = self.world_to_screen(df["x"], df["y"])
                p.drawLine(QPointF(sx1, sy1), QPointF(sxf, syf))

    def _draw_apf_warnings(self, p: QPainter):
        """Vẽ đường kẻ cảnh báo màu đỏ khi cự ly < 1.20m."""
        p.setPen(QPen(QColor(239, 68, 68, 220), 2, Qt.PenStyle.DashDotLine))
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))

        for id1, id2, dist in self.warning_pairs:
            d1 = self.render_poses.get(id1) or self.drones_data.get(id1)
            d2 = self.render_poses.get(id2) or self.drones_data.get(id2)
            if d1 and d2 and d1.get("connected", False) and d2.get("connected", False):
                sx1, sy1 = self.world_to_screen(d1["x"], d1["y"])
                sx2, sy2 = self.world_to_screen(d2["x"], d2["y"])

                p.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

                mid_x = (sx1 + sx2) / 2.0
                mid_y = (sy1 + sy2) / 2.0
                p.setPen(QPen(QColor(248, 113, 113)))
                p.drawText(QPointF(mid_x - 16, mid_y - 4), f"{dist:.2f}m")

    def _draw_trails(self, p: QPainter):
        """Vẽ vệt di chuyển thanh mảnh."""
        for d_id, points in self.trail_history.items():
            if len(points) < 2:
                continue
            color = DRONE_COLORS.get(d_id, {}).get("rgb", (200, 200, 200))
            pen = QPen(QColor(color[0], color[1], color[2], 60), 1)
            p.setPen(pen)

            pts_screen = [QPointF(*self.world_to_screen(wx, wy)) for wx, wy in points]
            for i in range(len(pts_screen) - 1):
                p.drawLine(pts_screen[i], pts_screen[i + 1])

    def _draw_drones(self, p: QPainter):
        """Vẽ 5 Drone bằng toạ độ nội suy 60 FPS (Sub-pixel antialiased)."""
        scale = self.scale_pixels_per_meter * self.zoom_factor

        for d_id in range(1, 6):
            d = self.render_poses.get(d_id) or self.drones_data.get(d_id)
            if not d:
                continue

            wx, wy, wz = d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)
            yaw_deg = d.get("yaw", 0.0)
            connected = d.get("connected", False)
            color_info = DRONE_COLORS.get(d_id, {"hex": "#FFFFFF", "rgb": (200, 200, 200)})
            rgb = color_info["rgb"]

            sx, sy = self.world_to_screen(wx, wy)

            if connected:
                # Vòng an toàn d_safe = 0.85m (chuẩn APF khoảng cách danh định 1.6m)
                r_safe_px = 0.85 * scale
                p.setPen(QPen(QColor(rgb[0], rgb[1], rgb[2], 50), 1, Qt.PenStyle.DotLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(sx, sy), r_safe_px, r_safe_px)

                # Vẽ thân Drone (Chevron tối giản)
                p.save()
                p.translate(sx, sy)
                rot_angle = -yaw_deg
                p.rotate(rot_angle)

                drone_poly = QPolygonF([
                    QPointF(10, 0),
                    QPointF(-8, -6),
                    QPointF(-4, 0),
                    QPointF(-8, 6),
                ])
                p.setPen(QPen(QColor(220, 220, 220), 1))
                p.setBrush(QBrush(QColor(rgb[0], rgb[1], rgb[2])))
                p.drawPolygon(drone_poly)

                p.setBrush(QBrush(QColor(255, 255, 255)))
                p.drawEllipse(QPointF(0, 0), 2, 2)
                p.restore()

                # Nhãn tên drone
                p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
                p.setPen(QPen(QColor(rgb[0], rgb[1], rgb[2])))
                label_text = f"D{d_id}" + ("*" if d_id == 1 else "")
                p.drawText(QPointF(sx + 10, sy - 3), label_text)

                p.setFont(QFont("Consolas", 7))
                p.setPen(QPen(QColor(150, 150, 150)))
                p.drawText(QPointF(sx + 10, sy + 8), f"Z:{wz:.1f}m")
            else:
                p.setPen(QPen(QColor(60, 60, 60), 1))
                p.setBrush(QBrush(QColor(30, 30, 30)))
                p.drawEllipse(QPointF(sx, sy), 4, 4)
                p.setFont(QFont("Consolas", 7))
                p.setPen(QPen(QColor(80, 80, 80)))
                p.drawText(QPointF(sx + 6, sy + 3), f"D{d_id}")

    def _draw_overlay(self, p: QPainter):
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        p.setPen(QPen(QColor(100, 100, 100)))
        p.drawText(10, 18, "RADAR 2D (TOP-DOWN)")

        # Thước đo cự ly góc dưới
        scale = self.scale_pixels_per_meter * self.zoom_factor
        bar_len_meters = 2.0
        bar_len_px = bar_len_meters * scale
        by = self.height() - 12
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawLine(10, by, int(10 + bar_len_px), by)
        p.drawLine(10, by - 3, 10, by + 3)
        p.drawLine(int(10 + bar_len_px), by - 3, int(10 + bar_len_px), by + 3)
        p.drawText(int(10 + bar_len_px / 2 - 6), by - 4, f"{bar_len_meters:.0f}m")
