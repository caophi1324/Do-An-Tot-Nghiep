"""Modes Dock widget allowing switching between Simple, Advanced, Enroll, Playback, and Monitor modes."""
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QDockWidget,
    QLabel,
)
from core.config import SystemMode


class ModesDock(QDockWidget):
    """OBS 'Scenes' mapped to System Operational Modes."""

    mode_changed = Signal(SystemMode)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Chế độ (Modes)", parent)
        self.setObjectName("ModesDock")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)

        self.modes_map = [
            (SystemMode.SIMPLE, "● Đơn giản (Simple)", "Tối ưu hóa cho người dùng mới, tự động cân bằng"),
            (SystemMode.ADVANCED, "⚙ Nâng cao (Advanced)", "Hiển thị đầy đủ thanh trượt tham số và landmark"),
            (SystemMode.ENROLL, "👤 Đăng ký (Enroll)", "Khung hướng dẫn chụp nhiều góc và chấm điểm chất lượng"),
            (SystemMode.PLAYBACK, "▶ Test / Playback", "Chạy kiểm thử với file video"),
            (SystemMode.MONITOR, "🛡 Giám sát (Monitor)", "Chế độ giám sát an ninh liên tục"),
        ]

        for mode, title, desc in self.modes_map:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, mode)
            item.setToolTip(desc)
            self.list_widget.addItem(item)

        self.list_widget.setCurrentRow(0)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)

        layout.addWidget(self.list_widget)
        self.setWidget(container)

    def _on_item_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is not None:
            mode = current.data(Qt.UserRole)
            self.mode_changed.emit(mode)

    def select_mode(self, mode: SystemMode) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.data(Qt.UserRole) == mode:
                self.list_widget.setCurrentRow(row)
                break
