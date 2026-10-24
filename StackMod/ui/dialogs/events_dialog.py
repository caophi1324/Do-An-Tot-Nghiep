"""Recognition Events Log dialog with table view, search, and CSV export."""
from __future__ import annotations

import csv
import os
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QLabel,
)
from db.repo import DatabaseRepo


class EventsDialog(QDialog):
    """Event log viewer showing historical recognition occurrences."""

    def __init__(self, repo: DatabaseRepo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.setWindowTitle("Lịch sử nhận diện (Recognition Events Log)")
        self.setMinimumSize(780, 480)

        self._init_ui()
        self.load_events()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top Bar (Search + Refresh + Export)
        top_bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Tìm theo tên hoặc mã đối tượng...")
        # Debounce: wait 400ms after last keystroke before searching
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(400)
        self._search_timer.timeout.connect(self._do_search)
        self.search_edit.textChanged.connect(lambda: self._search_timer.start())
        top_bar.addWidget(self.search_edit)

        self.btn_refresh = QPushButton("🔄 Làm mới")
        self.btn_export = QPushButton("📥 Xuất CSV")
        top_bar.addWidget(self.btn_refresh)
        top_bar.addWidget(self.btn_export)
        layout.addLayout(top_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID",
            "Thời gian",
            "Track #",
            "Đối tượng",
            "Mã",
            "Độ tương đồng",
            "Chất lượng",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

        self.lbl_status = QLabel("Hiển thị 0 sự kiện")
        self.lbl_status.setStyleSheet("color: #8f93a2; font-size: 8.5pt;")
        layout.addWidget(self.lbl_status)

        self.btn_refresh.clicked.connect(self.load_events)
        self.btn_export.clicked.connect(self._export_csv)

        self._events_cache = []

    def load_events(self) -> None:
        try:
            self._events_cache = self.repo.list_events(limit=200)
            self._populate_table(self._events_cache)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi tải dữ liệu", f"Không thể lấy lịch sử sự kiện: {e}")

    def _populate_table(self, events) -> None:
        self.table.setRowCount(len(events))
        for row, ev in enumerate(events):
            self.table.setItem(row, 0, QTableWidgetItem(str(ev["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(ev["ts"][:19].replace("T", " ")))
            self.table.setItem(row, 2, QTableWidgetItem(f"#{ev['track_id']}"))

            name_item = QTableWidgetItem(ev["person_name"])
            if ev["is_unknown"]:
                name_item.setForeground(Qt.darkYellow)
            else:
                name_item.setForeground(Qt.green)
            self.table.setItem(row, 3, name_item)

            self.table.setItem(row, 4, QTableWidgetItem(ev["person_code"]))

            sim_val = ev.get("similarity")
            sim_str = f"{sim_val * 100:.1f}%" if sim_val else "N/A"
            self.table.setItem(row, 5, QTableWidgetItem(sim_str))

            q_val = ev.get("quality")
            q_str = f"{q_val:.2f}" if q_val else "N/A"
            self.table.setItem(row, 6, QTableWidgetItem(q_str))

        self.lbl_status.setText(f"Hiển thị {len(events)} sự kiện")

    def _do_search(self) -> None:
        """Debounced search — queries DB directly instead of filtering in-memory cache."""
        q = self.search_edit.text().strip().lower()
        if not q:
            self.load_events()
            return
        try:
            # Search via DB for complete results
            all_events = self.repo.list_events(limit=200)
            filtered = [
                e for e in all_events
                if q in e["person_name"].lower() or q in e["person_code"].lower()
            ]
            self._populate_table(filtered)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi tìm kiếm", f"Không thể tìm kiếm: {e}")

    def _export_csv(self) -> None:
        if not self._events_cache:
            QMessageBox.information(self, "Thông báo", "Không có dữ liệu sự kiện để xuất!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất dữ liệu sự kiện CSV",
            "recognition_events.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Timestamp", "Track ID", "Person Name", "Person Code", "Similarity", "Quality", "Is Unknown"])
                for ev in self._events_cache:
                    writer.writerow([
                        ev["id"],
                        ev["ts"],
                        ev["track_id"],
                        ev["person_name"],
                        ev["person_code"],
                        ev["similarity"],
                        ev["quality"],
                        ev["is_unknown"],
                    ])
            QMessageBox.information(self, "Thành công", f"Đã xuất dữ liệu ra file:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất file", f"Không thể ghi file CSV: {e}")
