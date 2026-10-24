"""Controls Dock widget containing primary operational buttons."""
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDockWidget,
)


class ControlsDock(QDockWidget):
    """OBS 'Controls' panel: Main operational buttons."""

    start_clicked = Signal()
    stop_clicked = Signal()
    record_clicked = Signal()
    enroll_clicked = Signal()
    events_clicked = Signal()
    settings_clicked = Signal()
    reset_db_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Điều khiển (Controls)", parent)
        self.setObjectName("ControlsDock")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1. Start / Stop Button
        self.btn_run = QPushButton("▶ Bắt đầu nhận diện")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setMinimumHeight(34)

        # 2. Record Button
        self.btn_record = QPushButton("⏺ Ghi hình (REC)")
        self.btn_record.setMinimumHeight(30)

        # 3. Enroll Face Button
        self.btn_enroll = QPushButton("👤 Đăng ký khuôn mặt")
        self.btn_enroll.setMinimumHeight(30)

        # 4. Events Log Button
        self.btn_events = QPushButton("📋 Lịch sử sự kiện")
        self.btn_events.setMinimumHeight(30)

        # 5. Settings Button
        self.btn_settings = QPushButton("⚙ Cài đặt hệ thống")
        self.btn_settings.setMinimumHeight(30)

        # 6. Reset Database Button
        self.btn_reset_db = QPushButton("🗑️ Xóa cơ sở dữ liệu")
        self.btn_reset_db.setObjectName("btn_danger")
        self.btn_reset_db.setMinimumHeight(30)

        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_enroll)
        layout.addWidget(self.btn_events)
        layout.addWidget(self.btn_settings)
        layout.addWidget(self.btn_reset_db)
        layout.addStretch()

        self.setWidget(container)

        self._is_running = False
        self._is_recording = False

        self.btn_run.clicked.connect(self._toggle_run)
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_enroll.clicked.connect(self.enroll_clicked.emit)
        self.btn_events.clicked.connect(self.events_clicked.emit)
        self.btn_settings.clicked.connect(self.settings_clicked.emit)
        self.btn_reset_db.clicked.connect(self.reset_db_clicked.emit)

    def _toggle_run(self) -> None:
        if not self._is_running:
            self.start_clicked.emit()
        else:
            self.stop_clicked.emit()

    def set_running_state(self, running: bool) -> None:
        self._is_running = running
        if running:
            self.btn_run.setText("⏹ Dừng nhận diện")
            self.btn_run.setObjectName("btn_danger")
        else:
            self.btn_run.setText("▶ Bắt đầu nhận diện")
            self.btn_run.setObjectName("btn_primary")
        # Re-apply stylesheet
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

    def _toggle_record(self) -> None:
        self.record_clicked.emit()

    def set_recording_state(self, recording: bool) -> None:
        self._is_recording = recording
        if recording:
            self.btn_record.setText("⏹ Dừng ghi hình")
            self.btn_record.setObjectName("btn_danger")
        else:
            self.btn_record.setText("⏺ Ghi hình (REC)")
            self.btn_record.setObjectName("")
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)
