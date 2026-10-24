"""System Settings dialog for fine-grained configuration and database maintenance."""
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QMessageBox,
    QDialogButtonBox,
    QGroupBox,
)

from core.config import AppConfig, TrackingMode
from db.repo import DatabaseRepo
from core.recognizer import FaceRecognizer


class SettingsDialog(QDialog):
    """Settings dialog managing detection, tracking, thresholds, and FAISS index sync."""

    config_saved = Signal(AppConfig)

    def __init__(
        self,
        cfg: AppConfig,
        repo: DatabaseRepo,
        recognizer: FaceRecognizer,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.repo = repo
        self.recognizer = recognizer

        self.setWindowTitle("Cài đặt hệ thống (System Settings)")
        self.setMinimumSize(540, 420)

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # Tab 1: Detection & Tracking
        tab_det = QWidget()
        layout_det = QVBoxLayout(tab_det)

        grp_det = QGroupBox("Phát hiện người (YOLOv11n) & Mặt (RetinaFace)")
        l_det = QVBoxLayout(grp_det)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Ngưỡng tin cậy Person (Conf):"))
        self.spin_person_conf = QDoubleSpinBox()
        self.spin_person_conf.setRange(0.1, 0.99)
        self.spin_person_conf.setSingleStep(0.05)
        self.spin_person_conf.setValue(self.cfg.person_det.conf)
        h1.addWidget(self.spin_person_conf)
        l_det.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Ngưỡng tin cậy Face (Conf):"))
        self.spin_face_conf = QDoubleSpinBox()
        self.spin_face_conf.setRange(0.1, 0.99)
        self.spin_face_conf.setSingleStep(0.05)
        self.spin_face_conf.setValue(self.cfg.face_det.conf)
        h2.addWidget(self.spin_face_conf)
        l_det.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Tỉ lệ crop thân trên (Upper Ratio):"))
        self.spin_crop_ratio = QDoubleSpinBox()
        self.spin_crop_ratio.setRange(0.3, 0.9)
        self.spin_crop_ratio.setSingleStep(0.05)
        self.spin_crop_ratio.setValue(self.cfg.face_det.crop_upper_ratio)
        h3.addWidget(self.spin_crop_ratio)
        l_det.addLayout(h3)

        layout_det.addWidget(grp_det)

        grp_track = QGroupBox("Bám bắt đối tượng (ByteTrack)")
        l_track = QVBoxLayout(grp_track)

        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Chế độ Tracking:"))
        self.combo_track_mode = QComboBox()
        self.combo_track_mode.addItems(["Theo Người (Person)", "Theo Mặt (Face)"])
        self.combo_track_mode.setCurrentIndex(0 if self.cfg.tracking.mode == TrackingMode.PERSON else 1)
        h4.addWidget(self.combo_track_mode)
        l_track.addLayout(h4)

        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Track Buffer (frames):"))
        self.spin_track_buffer = QSpinBox()
        self.spin_track_buffer.setRange(5, 180)
        self.spin_track_buffer.setValue(self.cfg.tracking.track_buffer)
        h5.addWidget(self.spin_track_buffer)
        l_track.addLayout(h5)

        layout_det.addWidget(grp_track)
        layout_det.addStretch()
        tabs.addTab(tab_det, "Phát hiện & Tracking")

        # Tab 2: Recognition & FAISS
        tab_rec = QWidget()
        layout_rec = QVBoxLayout(tab_rec)

        grp_rec = QGroupBox("Nhận diện (EdgeFace + FAISS)")
        l_rec = QVBoxLayout(grp_rec)

        h6 = QHBoxLayout()
        h6.addWidget(QLabel("Ngưỡng so khớp Cosine:"))
        self.spin_match_thr = QDoubleSpinBox()
        self.spin_match_thr.setRange(0.1, 0.95)
        self.spin_match_thr.setSingleStep(0.05)
        self.spin_match_thr.setValue(self.cfg.recognition.match_thr)
        h6.addWidget(self.spin_match_thr)
        l_rec.addLayout(h6)

        h7 = QHBoxLayout()
        h7.addWidget(QLabel("Số phiếu bầu tối thiểu (Vote Min):"))
        self.spin_vote_min = QSpinBox()
        self.spin_vote_min.setRange(1, 15)
        self.spin_vote_min.setValue(self.cfg.recognition.vote_min)
        h7.addWidget(self.spin_vote_min)
        l_rec.addLayout(h7)

        h8 = QHBoxLayout()
        h8.addWidget(QLabel("Cửa sổ bình chọn (Vote Window):"))
        self.spin_vote_win = QSpinBox()
        self.spin_vote_win.setRange(3, 30)
        self.spin_vote_win.setValue(self.cfg.recognition.vote_window)
        h8.addWidget(self.spin_vote_win)
        l_rec.addLayout(h8)

        layout_rec.addWidget(grp_rec)

        grp_sync = QGroupBox("Bảo trì Vector Database FAISS")
        l_sync = QVBoxLayout(grp_sync)
        self.lbl_faiss_count = QLabel(f"Tổng số vector đang nạp: {self.recognizer.total_templates}")
        l_sync.addWidget(self.lbl_faiss_count)

        self.btn_rebuild_faiss = QPushButton("🔄 Đồng bộ lại FAISS từ Database")
        l_sync.addWidget(self.btn_rebuild_faiss)
        layout_rec.addWidget(grp_sync)

        layout_rec.addStretch()
        tabs.addTab(tab_rec, "Nhận diện & FAISS")

        # Tab 3: Storage Paths
        tab_storage = QWidget()
        layout_storage = QVBoxLayout(tab_storage)

        layout_storage.addWidget(QLabel("Đường dẫn Database SQLite:"))
        self.edit_db_path = QLineEdit(self.cfg.storage.db_url)
        layout_storage.addWidget(self.edit_db_path)

        layout_storage.addWidget(QLabel("Thư mục lưu Vector FAISS:"))
        self.edit_faiss_path = QLineEdit(self.cfg.storage.faiss_path)
        layout_storage.addWidget(self.edit_faiss_path)

        layout_storage.addWidget(QLabel("Thư mục lưu Snapshots:"))
        self.edit_snap_path = QLineEdit(self.cfg.storage.snapshots_dir)
        layout_storage.addWidget(self.edit_snap_path)

        layout_storage.addStretch()
        tabs.addTab(tab_storage, "Lưu trữ & Thư mục")

        # Tab 4: Hardware & GPU
        tab_hw = QWidget()
        layout_hw = QVBoxLayout(tab_hw)

        grp_hw = QGroupBox("Tăng tốc phần cứng AI (Hardware Acceleration)")
        l_hw = QVBoxLayout(grp_hw)

        h_hw1 = QHBoxLayout()
        h_hw1.addWidget(QLabel("Môi trường thực thi (Provider):"))
        self.combo_provider = QComboBox()
        self.combo_provider.addItems([
            "Tự động nhận diện (Auto Detect)",
            "DirectML (AMD Radeon / Intel / Windows GPU)",
            "NVIDIA CUDA",
            "CPU thuần (CPU Execution)",
        ])

        # Map current setting
        prov = str(self.cfg.hardware.provider).lower()
        if prov == "directml":
            self.combo_provider.setCurrentIndex(1)
        elif prov == "cuda":
            self.combo_provider.setCurrentIndex(2)
        elif prov == "cpu":
            self.combo_provider.setCurrentIndex(3)
        else:
            self.combo_provider.setCurrentIndex(0)

        h_hw1.addWidget(self.combo_provider)
        l_hw.addLayout(h_hw1)

        import onnxruntime as ort
        avail_list = ", ".join(ort.get_available_providers())
        self.lbl_avail_hw = QLabel(f"Các thiết bị phần cứng được phát hiện trên máy:\n{avail_list}")
        self.lbl_avail_hw.setStyleSheet("color: #2ea043; font-size: 8.5pt; margin-top: 6px;")
        l_hw.addWidget(self.lbl_avail_hw)

        layout_hw.addWidget(grp_hw)
        layout_hw.addStretch()
        tabs.addTab(tab_hw, "Phần cứng & GPU")

        main_layout.addWidget(tabs)

        # Buttons
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        main_layout.addWidget(self.btn_box)

        self.btn_rebuild_faiss.clicked.connect(self._on_rebuild_faiss)
        self.btn_box.accepted.connect(self._on_save)
        self.btn_box.rejected.connect(self.reject)

    def _on_rebuild_faiss(self) -> None:
        try:
            count = self.repo.sync_faiss(self.recognizer)
            self.lbl_faiss_count.setText(f"Tổng số vector đang nạp: {count}")
            QMessageBox.information(self, "Thành công", f"Đã đồng bộ {count} mẫu khuôn mặt vào chỉ mục FAISS!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể đồng bộ FAISS: {e}")

    def _on_save(self) -> None:
        self.cfg.person_det.conf = float(self.spin_person_conf.value())
        self.cfg.face_det.conf = float(self.spin_face_conf.value())
        self.cfg.face_det.crop_upper_ratio = float(self.spin_crop_ratio.value())
        self.cfg.tracking.mode = TrackingMode.PERSON if self.combo_track_mode.currentIndex() == 0 else TrackingMode.FACE
        self.cfg.tracking.track_buffer = int(self.spin_track_buffer.value())
        self.cfg.recognition.match_thr = float(self.spin_match_thr.value())
        self.cfg.recognition.vote_min = int(self.spin_vote_min.value())
        self.cfg.recognition.vote_window = int(self.spin_vote_win.value())
        # Hardware provider
        hw_idx = self.combo_provider.currentIndex()
        from core.config import ExecutionProvider
        if hw_idx == 1:
            self.cfg.hardware.provider = ExecutionProvider.DIRECTML
        elif hw_idx == 2:
            self.cfg.hardware.provider = ExecutionProvider.CUDA
        elif hw_idx == 3:
            self.cfg.hardware.provider = ExecutionProvider.CPU
        else:
            self.cfg.hardware.provider = ExecutionProvider.AUTO

        self.config_saved.emit(self.cfg)
        self.accept()
