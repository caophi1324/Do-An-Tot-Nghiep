"""Overlay Dock widget for toggling visual annotation layers."""
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QCheckBox,
    QDockWidget,
)
from core.config import OverlayConfig


class OverlayDock(QDockWidget):
    """OBS 'Scene Transitions' analog: Visual overlay layer toggles."""

    overlay_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Hiển thị (Overlay)", parent)
        self.setObjectName("OverlayDock")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.cb_person_box = QCheckBox("Bbox Người (Person)")
        self.cb_person_box.setChecked(True)

        self.cb_face_box = QCheckBox("Bbox Mặt (Face)")
        self.cb_face_box.setChecked(True)

        self.cb_landmarks = QCheckBox("5 Landmark khuôn mặt")
        self.cb_landmarks.setChecked(False)

        self.cb_id = QCheckBox("Track ID (#)")
        self.cb_id.setChecked(True)

        self.cb_name = QCheckBox("Tên danh tính")
        self.cb_name.setChecked(True)

        self.cb_quality = QCheckBox("Điểm Face Quality")
        self.cb_quality.setChecked(True)

        self.cb_hud = QCheckBox("Chỉ số FPS / Độ trễ")
        self.cb_hud.setChecked(True)

        layout.addWidget(self.cb_person_box)
        layout.addWidget(self.cb_face_box)
        layout.addWidget(self.cb_landmarks)
        layout.addWidget(self.cb_id)
        layout.addWidget(self.cb_name)
        layout.addWidget(self.cb_quality)
        layout.addWidget(self.cb_hud)
        layout.addStretch()

        self.setWidget(container)

        self.cb_person_box.stateChanged.connect(self._emit_change)
        self.cb_face_box.stateChanged.connect(self._emit_change)
        self.cb_landmarks.stateChanged.connect(self._emit_change)
        self.cb_id.stateChanged.connect(self._emit_change)
        self.cb_name.stateChanged.connect(self._emit_change)
        self.cb_quality.stateChanged.connect(self._emit_change)
        self.cb_hud.stateChanged.connect(self._emit_change)

    def _emit_change(self) -> None:
        self.overlay_changed.emit({
            "show_person_bbox": self.cb_person_box.isChecked(),
            "show_face_bbox": self.cb_face_box.isChecked(),
            "show_landmarks": self.cb_landmarks.isChecked(),
            "show_id": self.cb_id.isChecked(),
            "show_name": self.cb_name.isChecked(),
            "show_quality": self.cb_quality.isChecked(),
            "show_fps": self.cb_hud.isChecked(),
            "show_latency": self.cb_hud.isChecked(),
        })

    def apply_config(self, cfg: OverlayConfig) -> None:
        self.cb_person_box.setChecked(cfg.show_person_bbox)
        self.cb_face_box.setChecked(cfg.show_face_bbox)
        self.cb_landmarks.setChecked(cfg.show_landmarks)
        self.cb_id.setChecked(cfg.show_id)
        self.cb_name.setChecked(cfg.show_name)
        self.cb_quality.setChecked(cfg.show_quality)
        self.cb_hud.setChecked(cfg.show_fps)
