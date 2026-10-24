"""Interactive Face Enrollment Dialog with multi-pose capture and quality filtering."""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QComboBox,
)

from core.recognizer import FaceRecognizer
from core.quality import FaceQualityEstimator, filter_5_landmarks
from core.detectors import FaceDetector
from core.config import AppConfig
from db.repo import DatabaseRepo


class EnrollDialog(QDialog):
    """Face Enrollment dialog capturing multiple angles and filtering by eDifFIQA quality."""

    face_enrolled = Signal(str, str)  # (code, name)
    _eval_result_ready = Signal(object, object, float, str, str)  # (bbox, landmarks, quality, guide_text, guide_style)

    def __init__(
        self,
        repo: DatabaseRepo,
        recognizer: FaceRecognizer,
        face_det: FaceDetector,
        quality_est: FaceQualityEstimator,
        cfg: AppConfig,
        parent: Optional[QWidget] = None,
        inference_lock: Optional[threading.RLock] = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self.recognizer = recognizer
        self.face_det = face_det
        self.quality_est = quality_est
        self.cfg = cfg
        self.inference_lock = inference_lock or threading.RLock()

        self.setWindowTitle("Đăng ký khuôn mặt mới (Face Enrollment)")
        self.setMinimumSize(680, 520)

        # Captured templates: list of (embedding, quality, aligned_crop_bgr)
        self.captured_templates: List[Tuple[np.ndarray, float, np.ndarray]] = []
        self._current_frame: Optional[np.ndarray] = None

        # Throttling & worker state (~5 Hz)
        self._last_eval_time: float = 0.0
        self._worker_busy: bool = False
        self._latest_face_box: Optional[Tuple[int, int, int, int]] = None
        self._latest_landmarks: Optional[np.ndarray] = None

        self._eval_result_ready.connect(self._on_eval_result)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(12)

        # Left Column: Camera Preview & Instructions
        left_layout = QVBoxLayout()

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(360, 270)
        self.preview_label.setStyleSheet("background-color: #0d0e14; border: 1px solid #36384a; border-radius: 6px;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Camera Feed")
        left_layout.addWidget(self.preview_label)

        self.guide_label = QLabel("Hướng dẫn: Nhìn thẳng vào camera, giữ mặt trong khung hình.")
        self.guide_label.setStyleSheet("color: #2b78e4; font-weight: bold; font-size: 9pt;")
        self.guide_label.setWordWrap(True)
        left_layout.addWidget(self.guide_label)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setFormat("Chất lượng: %v%")
        left_layout.addWidget(self.quality_bar)

        left_btn_layout = QHBoxLayout()
        self.capture_btn = QPushButton("📸 Chụp mẫu (Capture)")
        self.capture_btn.setObjectName("btn_primary")
        self.import_btn = QPushButton("📁 Nhập từ ảnh...")
        left_btn_layout.addWidget(self.capture_btn)
        left_btn_layout.addWidget(self.import_btn)
        left_layout.addLayout(left_btn_layout)

        main_layout.addLayout(left_layout, 1)

        # Right Column: Metadata & Thumbnails
        right_layout = QVBoxLayout()

        meta_group = QGroupBox("Thông tin đối tượng")
        meta_layout = QVBoxLayout(meta_group)

        meta_layout.addWidget(QLabel("Mã đối tượng (Code / ID):"))
        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("VD: NV001")
        meta_layout.addWidget(self.code_edit)

        meta_layout.addWidget(QLabel("Họ và tên (Full Name):"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("VD: Nguyễn Văn A")
        meta_layout.addWidget(self.name_edit)

        meta_layout.addWidget(QLabel("Nhóm (Group):"))
        self.group_combo = QComboBox()
        self.group_combo.setEditable(True)
        self.group_combo.addItems(["General", "Employees", "VIP", "Guests"])
        meta_layout.addWidget(self.group_combo)

        right_layout.addWidget(meta_group)

        # Thumbnails strip
        thumb_group = QGroupBox("Các mẫu đã lưu (Tối đa 5 mẫu)")
        self.thumb_layout = QHBoxLayout(thumb_group)
        self.thumb_labels: List[QLabel] = []
        for i in range(5):
            lbl = QLabel()
            lbl.setFixedSize(56, 56)
            lbl.setStyleSheet("background-color: #181922; border: 1px dashed #3b3f54; border-radius: 4px;")
            lbl.setAlignment(Qt.AlignCenter)
            self.thumb_labels.append(lbl)
            self.thumb_layout.addWidget(lbl)
        right_layout.addWidget(thumb_group)

        # Save & Cancel Buttons
        btn_box = QHBoxLayout()
        self.save_btn = QPushButton("💾 Hoàn tất & Lưu vào Database")
        self.save_btn.setObjectName("btn_success")
        self.save_btn.setEnabled(False)

        self.cancel_btn = QPushButton("Hủy bỏ")
        btn_box.addWidget(self.save_btn)
        btn_box.addWidget(self.cancel_btn)
        right_layout.addLayout(btn_box)

        main_layout.addLayout(right_layout, 1)

        self.capture_btn.clicked.connect(self._capture_current_frame)
        self.import_btn.clicked.connect(self._import_from_file)
        self.save_btn.clicked.connect(self._save_enrollment)
        self.cancel_btn.clicked.connect(self.reject)

    @Slot(object, object, float, str, str)
    def _on_eval_result(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        landmarks: Optional[np.ndarray],
        quality: float,
        guide_text: str,
        guide_style: str,
    ) -> None:
        self._latest_face_box = bbox
        self._latest_landmarks = landmarks
        self.quality_bar.setValue(int(quality * 100))
        self.guide_label.setText(guide_text)
        self.guide_label.setStyleSheet(guide_style)
        self._worker_busy = False

    def _run_eval_worker(self, frame_bgr: np.ndarray) -> None:
        try:
            with self.inference_lock:
                faces = self.face_det.detect_in_frame(frame_bgr)
                if faces:
                    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    q = 0.0
                    landmarks_copy = None
                    if largest.landmarks is not None:
                        q = float(self.quality_est.score(frame_bgr, largest.landmarks))
                        landmarks_copy = np.array(largest.landmarks, copy=True)

                    if q >= self.cfg.quality.min_quality:
                        guide_text = "✅ Chất lượng khuôn mặt tốt! Sẵn sàng chụp."
                        guide_style = "color: #2ea043; font-weight: bold;"
                    else:
                        guide_text = "⚠️ Chất lượng thấp: Vui lòng nhìn thẳng và giữ nét."
                        guide_style = "color: #e54d4f; font-weight: bold;"

                    b = largest.bbox
                    box_tuple = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    self._eval_result_ready.emit(box_tuple, landmarks_copy, q, guide_text, guide_style)
                else:
                    self._eval_result_ready.emit(None, None, 0.0, "Chưa tìm thấy khuôn mặt trong khung hình.", "color: #b0b4c3;")
        except Exception as e:
            print(f"[EnrollDialog] Eval worker error: {e}")
            self._worker_busy = False

    def set_live_frame(self, frame_bgr: np.ndarray) -> None:
        """Called by main window when a new video frame arrives."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        self._current_frame = frame_bgr.copy()

        now = time.time()
        # Throttled to ~5 Hz (every 0.20s), offloaded to worker thread
        if (now - self._last_eval_time >= 0.20) and not self._worker_busy:
            self._last_eval_time = now
            self._worker_busy = True
            eval_copy = self._current_frame.copy()
            worker_thread = threading.Thread(target=self._run_eval_worker, args=(eval_copy,), daemon=True)
            worker_thread.start()

        # Render display on GUI thread using cached bbox / landmarks
        display = self._current_frame.copy()
        if self._latest_face_box is not None:
            x1, y1, x2, y2 = self._latest_face_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (43, 120, 228), 2)
        if self._latest_landmarks is not None:
            for lx, ly in np.asarray(self._latest_landmarks, dtype=int):
                cv2.circle(display, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        h, w, ch = display.shape
        q_img = QImage(display.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def _process_and_add_face(self, frame_bgr: np.ndarray) -> bool:
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        with self.inference_lock:
            faces = self.face_det.detect_in_frame(frame_bgr)
            if not faces:
                QMessageBox.warning(self, "Không tìm thấy mặt", "Không phát hiện khuôn mặt nào trong ảnh!")
                return False

            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if largest.landmarks is None:
                QMessageBox.warning(self, "Lỗi landmark", "Không tìm đủ 5 điểm landmark khuôn mặt!")
                return False

            # Filter landmarks
            passed, reason, _ = filter_5_landmarks(
                landmarks=largest.landmarks,
                face_bbox=largest.bbox,
                face_conf=largest.conf,
                cfg=self.cfg.landmark_filter,
            )
            if not passed:
                QMessageBox.warning(self, "Chất lượng góc chụp kém", f"Mẫu không đạt tiêu chuẩn: {reason}")
                return False

            # Align to 112x112
            aligned, _ = self.recognizer.align(frame_bgr, largest.landmarks)
            q = self.quality_est.score(frame_bgr, largest.landmarks)
            emb = self.recognizer.extract_embedding(aligned)

        # Add to captured list (up to 5)
        if len(self.captured_templates) >= 5:
            # Replace lowest quality template
            min_idx = min(range(len(self.captured_templates)), key=lambda i: self.captured_templates[i][1])
            if q > self.captured_templates[min_idx][1]:
                self.captured_templates[min_idx] = (emb, q, aligned)
        else:
            self.captured_templates.append((emb, q, aligned))

        self._update_thumbnails()
        self.save_btn.setEnabled(len(self.captured_templates) > 0)
        return True

    def _capture_current_frame(self) -> None:
        if self._current_frame is not None:
            self._process_and_add_face(self._current_frame)
        else:
            QMessageBox.warning(self, "Chưa có hình", "Chưa nhận được tín hiệu hình ảnh từ camera!")

    def _import_from_file(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh khuôn mặt",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*.*)",
        )
        if file_paths:
            for p in file_paths:
                img = None
                try:
                    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    pass
                if img is None:
                    img = cv2.imread(p)
                if img is not None:
                    self._process_and_add_face(img)

    def _update_thumbnails(self) -> None:
        for i, lbl in enumerate(self.thumb_labels):
            if i < len(self.captured_templates):
                _, q, aligned = self.captured_templates[i]
                h, w, ch = aligned.shape
                q_img = QImage(aligned.data, w, h, ch * w, QImage.Format_BGR888)
                pix = QPixmap.fromImage(q_img).scaled(
                    lbl.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                lbl.setPixmap(pix)
                lbl.setToolTip(f"Mẫu #{i+1} - Chất lượng: {q:.2f}")
            else:
                lbl.clear()
                lbl.setText(f"Mẫu {i+1}")
                lbl.setToolTip("")

    def _save_enrollment(self) -> None:
        code = self.code_edit.text().strip()
        name = self.name_edit.text().strip()
        group = self.group_combo.currentText().strip() or "General"

        if not code:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Mã đối tượng!")
            return
        if not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Họ và tên!")
            return
        if not self.captured_templates:
            QMessageBox.warning(self, "Thiếu mẫu", "Chưa có mẫu khuôn mặt nào được chụp!")
            return

        try:
            person = self.repo.enroll_face(
                code=code,
                full_name=name,
                group_name=group,
                templates=self.captured_templates,
                recognizer=self.recognizer,
                snapshots_dir=self.cfg.storage.snapshots_dir,
            )
            person_name = getattr(person, "full_name", name)
            person_code = getattr(person, "code", code)
            QMessageBox.information(
                self,
                "Đăng ký thành công",
                f"Đã lưu đối tượng '{person_name}' (#{person_code}) với {len(self.captured_templates)} mẫu khuôn mặt!",
            )
            self.face_enrolled.emit(str(person_code), str(person_name))
            self.accept()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Lỗi đăng ký", f"Không thể lưu vào database: {e}")
