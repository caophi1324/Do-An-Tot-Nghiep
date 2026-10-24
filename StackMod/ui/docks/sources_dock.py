"""Sources Dock widget managing camera and video input sources."""
from __future__ import annotations

from typing import Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QDockWidget,
    QInputDialog,
    QFileDialog,
    QMessageBox,
    QDialog,
    QLabel,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
)


from core.sources import normalize_stream_uri


class AddSourceDialog(QDialog):
    """Dialog for creating a new video source."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thêm nguồn video (Add Source)")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Source Type
        layout.addWidget(QLabel("Loại nguồn (Source Type):"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Webcam (USB)",
            "DroidCam (HTTP MJPEG)",
            "RTSP Stream (IP Camera)",
            "Tệp Video (Video File)",
        ])
        layout.addWidget(self.type_combo)

        # Name
        layout.addWidget(QLabel("Tên nguồn (Name):"))
        self.name_edit = QLineEdit("Camera 1")
        layout.addWidget(self.name_edit)

        # URI
        layout.addWidget(QLabel("Đường dẫn / URI:"))
        self.uri_layout = QHBoxLayout()
        self.uri_edit = QLineEdit("0")
        self.uri_layout.addWidget(self.uri_edit)
        self.browse_btn = QPushButton("Chọn tệp...")
        self.browse_btn.setVisible(False)
        self.uri_layout.addWidget(self.browse_btn)
        layout.addLayout(self.uri_layout)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.browse_btn.clicked.connect(self._on_browse_file)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _on_type_changed(self, idx: int) -> None:
        if idx == 0:  # Webcam
            self.uri_edit.setText("0")
            self.browse_btn.setVisible(False)
        elif idx == 1:  # DroidCam
            self.uri_edit.setText("http://192.168.1.100:4747/video")
            self.browse_btn.setVisible(False)
        elif idx == 2:  # RTSP
            self.uri_edit.setText("rtsp://admin:password@192.168.1.50:554/stream1")
            self.browse_btn.setVisible(False)
        elif idx == 3:  # Video file
            self.uri_edit.setText("")
            self.browse_btn.setVisible(True)

    def _on_browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.flv);;All Files (*.*)",
        )
        if file_path:
            self.uri_edit.setText(file_path)

    def get_source_data(self) -> Dict[str, str]:
        type_str = "webcam"
        idx = self.type_combo.currentIndex()
        if idx == 1:
            type_str = "droidcam"
        elif idx == 2:
            type_str = "rtsp"
        elif idx == 3:
            type_str = "file"

        uri = self.uri_edit.text().strip()
        if type_str in ("droidcam", "rtsp", "stream"):
            uri = normalize_stream_uri(uri)

        return {
            "name": self.name_edit.text().strip() or "Source",
            "type": type_str,
            "uri": uri,
        }


class SourcesDock(QDockWidget):
    """OBS 'Sources' dock managing camera inputs."""

    source_selected = Signal(str, str)  # (source_type, uri)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Nguồn (Sources)", parent)
        self.setObjectName("SourcesDock")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # Toolbar (+, -, Gear, Up, Down)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.add_btn = QPushButton("+")
        self.add_btn.setToolTip("Thêm nguồn mới")
        self.add_btn.setFixedWidth(28)

        self.remove_btn = QPushButton("−")
        self.remove_btn.setToolTip("Xóa nguồn đã chọn")
        self.remove_btn.setFixedWidth(28)

        self.up_btn = QPushButton("▲")
        self.up_btn.setToolTip("Di chuyển lên")
        self.up_btn.setFixedWidth(28)

        self.down_btn = QPushButton("▼")
        self.down_btn.setToolTip("Di chuyển xuống")
        self.down_btn.setFixedWidth(28)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.up_btn)
        btn_layout.addWidget(self.down_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        self.setWidget(container)

        self.add_btn.clicked.connect(self._add_source)
        self.remove_btn.clicked.connect(self._remove_source)
        self.up_btn.clicked.connect(self._move_up)
        self.down_btn.clicked.connect(self._move_down)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)

        # Add default items
        self.add_source_item("Webcam 0", "webcam", "0")
        self.list_widget.setCurrentRow(0)

    def add_source_item(self, name: str, source_type: str, uri: str) -> None:
        item = QListWidgetItem(f"🎥 {name} ({source_type})")
        item.setData(Qt.UserRole, {"name": name, "type": source_type, "uri": uri})
        self.list_widget.addItem(item)

    def _add_source(self) -> None:
        dialog = AddSourceDialog(self)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_source_data()
            self.add_source_item(data["name"], data["type"], data["uri"])
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def _remove_source(self) -> None:
        row = self.list_widget.currentRow()
        if row >= 0:
            if self.list_widget.count() == 1:
                QMessageBox.warning(self, "Cảnh báo", "Phải giữ ít nhất 1 nguồn video!")
                return
            self.list_widget.blockSignals(True)
            try:
                self.list_widget.takeItem(row)
            finally:
                self.list_widget.blockSignals(False)

    def _move_up(self) -> None:
        row = self.list_widget.currentRow()
        if row > 0:
            self.list_widget.blockSignals(True)
            try:
                item = self.list_widget.takeItem(row)
                self.list_widget.insertItem(row - 1, item)
                self.list_widget.setCurrentRow(row - 1)
            finally:
                self.list_widget.blockSignals(False)

    def _move_down(self) -> None:
        row = self.list_widget.currentRow()
        if row < self.list_widget.count() - 1 and row >= 0:
            self.list_widget.blockSignals(True)
            try:
                item = self.list_widget.takeItem(row)
                self.list_widget.insertItem(row + 1, item)
                self.list_widget.setCurrentRow(row + 1)
            finally:
                self.list_widget.blockSignals(False)

    def _on_item_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is not None:
            data = current.data(Qt.UserRole)
            if data:
                self.source_selected.emit(data["type"], data["uri"])

    def get_current_source(self) -> Optional[Dict[str, str]]:
        item = self.list_widget.currentItem()
        if item is not None:
            return item.data(Qt.UserRole)
        return None
