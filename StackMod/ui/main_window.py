"""OBS Studio-styled Main Application Window."""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional
import psutil
import numpy as np

from PySide6.QtCore import Qt, QTimer, QSettings, Signal, Slot, QMetaObject, Q_ARG, QThreadPool, QRunnable
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QStatusBar,
)

from core.config import AppConfig, SystemMode, TrackingMode
from core.runner import PipelineRunner
from core.sources import open_source, VideoSource
from db.repo import DatabaseRepo, get_db_repo

from .video_widget import VideoWidget
from .docks.modes_dock import ModesDock
from .docks.sources_dock import SourcesDock
from .docks.tuning_dock import TuningDock
from .docks.overlay_dock import OverlayDock
from .docks.controls_dock import ControlsDock

from .dialogs.enroll_dialog import EnrollDialog
from .dialogs.settings_dialog import SettingsDialog
from .dialogs.events_dialog import EventsDialog


class EnrollWorker(QRunnable):
    """Worker task extracting face embedding and enrolling person off GUI thread."""

    def __init__(self, main_win: MainWindow, raw_frame: np.ndarray, target_res, target_tid: int) -> None:
        super().__init__()
        self.main_win = main_win
        self.raw_frame = raw_frame
        self.target_res = target_res
        self.target_tid = target_tid

    def run(self) -> None:
        try:
            self.main_win._async_sample_and_enroll(self.raw_frame, self.target_res, self.target_tid)
        except Exception as e:
            print(f"[EnrollWorker] Error: {e}")


class MainWindow(QMainWindow):
    """OBS Studio style main application window for face tracking & recognition."""

    # Signal emitted from worker thread, connected to slot on main thread
    _frame_ready = Signal(np.ndarray, list, float, float)
    _auto_enroll_success = Signal(int, int, str)  # (target_tid, assigned_id, person_name)
    _auto_enroll_failed = Signal(int, str)  # (target_tid, message)

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AppConfig()
        self.cfg_path = os.path.join("data", "config.json")
        self.repo = get_db_repo(self.cfg.storage.db_url)

        self.setWindowTitle("AI Face Recognition & Tracking Studio (OBS Style)")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        # Apply dark theme
        self._load_theme()

        # Database session tracking
        self._current_session_id: Optional[int] = None
        self._active_track_identities: dict[int, Optional[int]] = {}

        # DB event logging queue (offload DB writes from frame callback)
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._db_writer_stop = threading.Event()
        self._db_writer_thread = threading.Thread(target=self._db_writer_loop, daemon=True)
        self._db_writer_thread.start()

        # Connect signal to slot (thread-safe UI update)
        self._frame_ready.connect(self._handle_frame_on_main_thread)
        self._ui_busy = threading.Event()

        # Pipeline Runner — callback emits signal instead of touching UI directly
        self.runner = PipelineRunner(
            config=self.cfg,
            on_frame_processed=self._on_frame_processed_from_worker,
            on_template_learned=self._on_template_learned,
        )

        # Timers
        self._run_start_time: Optional[float] = None
        self._rec_start_time: Optional[float] = None

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_status_bar)
        self._ui_timer.start(500)

        # Dialog instances
        self._enroll_dialog: Optional[EnrollDialog] = None

        self._init_ui()
        self._init_menu()
        self._init_status_bar()
        self._connect_signals()
        self._restore_dock_state()

    def _load_theme(self) -> None:
        qss_path = os.path.join(os.path.dirname(__file__), "theme.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _init_ui(self) -> None:
        # Central Video Viewport
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Docks
        self.modes_dock = ModesDock(self)
        self.sources_dock = SourcesDock(self)
        self.tuning_dock = TuningDock(self, cfg=self.cfg)
        self.overlay_dock = OverlayDock(self)
        self.controls_dock = ControlsDock(self)
        self.overlay_dock.apply_config(self.cfg.overlay)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.modes_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.sources_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.tuning_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.overlay_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.controls_dock)

        # Sync FAISS vector index with DB on launch
        try:
            self.repo.sync_faiss(self.runner.pipeline.recognizer)
        except Exception as e:
            print(f"[MainWindow] Warning: FAISS sync failed: {e}")

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("Tệp (File)")
        act_exit = QAction("Thoát", self)
        act_exit.setShortcut("Alt+F4")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # View Menu
        menu_view = menubar.addMenu("Xem (View)")
        menu_view.addAction(self.modes_dock.toggleViewAction())
        menu_view.addAction(self.sources_dock.toggleViewAction())
        menu_view.addAction(self.tuning_dock.toggleViewAction())
        menu_view.addAction(self.overlay_dock.toggleViewAction())
        menu_view.addAction(self.controls_dock.toggleViewAction())

        act_fullscreen = QAction("Toàn màn hình (Fullscreen)", self)
        act_fullscreen.setShortcut("F11")
        act_fullscreen.triggered.connect(self._toggle_fullscreen)
        menu_view.addSeparator()
        menu_view.addAction(act_fullscreen)

        # Profile Menu
        menu_profile = menubar.addMenu("Profile")
        act_prof_simple = QAction("Preset: Đơn giản", self)
        act_prof_simple.triggered.connect(lambda: self.modes_dock.select_mode(SystemMode.SIMPLE))
        act_prof_adv = QAction("Preset: Nâng cao", self)
        act_prof_adv.triggered.connect(lambda: self.modes_dock.select_mode(SystemMode.ADVANCED))
        menu_profile.addAction(act_prof_simple)
        menu_profile.addAction(act_prof_adv)

        # Tools Menu
        menu_tools = menubar.addMenu("Công cụ (Tools)")
        act_enroll = QAction("Đăng ký khuôn mặt... (E)", self)
        act_enroll.setShortcut("E")
        act_enroll.triggered.connect(self._open_enroll_dialog)
        act_events = QAction("Lịch sử nhận diện...", self)
        act_events.triggered.connect(self._open_events_dialog)
        act_settings = QAction("Cài đặt hệ thống...", self)
        act_settings.triggered.connect(self._open_settings_dialog)
        act_reset_db = QAction("🗑️ Xóa toàn bộ cơ sở dữ liệu...", self)
        act_reset_db.triggered.connect(self._confirm_and_reset_database)
        menu_tools.addAction(act_enroll)
        menu_tools.addAction(act_events)
        menu_tools.addAction(act_settings)
        menu_tools.addSeparator()
        menu_tools.addAction(act_reset_db)

        # Help Menu
        menu_help = menubar.addMenu("Trợ giúp (Help)")
        act_about = QAction("Giới thiệu (About)", self)
        act_about.triggered.connect(self._show_about)
        menu_help.addAction(act_about)

    def _init_status_bar(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.lbl_run_timer = QLabel("RUN 00:00:00")
        self.lbl_rec_timer = QLabel("REC 00:00:00")
        self.lbl_fps = QLabel("FPS 0.0")
        self.lbl_perf = QLabel("CPU 0% | Latency 0ms")
        self.lbl_tracks = QLabel("Tracks: 0")
        self.lbl_status = QLabel("● CHỜ SẴN")

        self.lbl_run_timer.setStyleSheet("color: #2b78e4; font-weight: bold; margin-right: 8px;")
        self.lbl_rec_timer.setStyleSheet("color: #777b8f; font-weight: bold; margin-right: 8px;")
        self.lbl_fps.setStyleSheet("color: #2ea043; font-weight: bold; margin-right: 8px;")
        self.lbl_perf.setStyleSheet("color: #b0b4c3; margin-right: 8px;")
        self.lbl_tracks.setStyleSheet("color: #ffffff; font-weight: bold; margin-right: 8px;")
        self.lbl_status.setStyleSheet("color: #8f93a2; font-weight: bold;")

        self.status_bar.addWidget(self.lbl_run_timer)
        self.status_bar.addWidget(self.lbl_rec_timer)
        self.status_bar.addWidget(self.lbl_fps)
        self.status_bar.addWidget(self.lbl_perf)
        self.status_bar.addWidget(self.lbl_tracks)
        self.status_bar.addPermanentWidget(self.lbl_status)

    def _connect_signals(self) -> None:
        # Controls signals
        self.controls_dock.start_clicked.connect(self.start_pipeline)
        self.controls_dock.stop_clicked.connect(self.stop_pipeline)
        self.controls_dock.record_clicked.connect(self.toggle_recording)
        self.controls_dock.enroll_clicked.connect(self._open_enroll_dialog)
        self.controls_dock.events_clicked.connect(self._open_events_dialog)
        self.controls_dock.settings_clicked.connect(self._open_settings_dialog)
        self.controls_dock.reset_db_clicked.connect(self._confirm_and_reset_database)

        # Modes signal
        self.modes_dock.mode_changed.connect(self._on_mode_changed)

        # Sources signal
        self.sources_dock.source_selected.connect(self._on_source_selected)

        # Tuning signal
        self.tuning_dock.tuning_changed.connect(self._on_tuning_changed)

        # Overlay signal
        self.overlay_dock.overlay_changed.connect(self._on_overlay_changed)

        # Video viewport click interaction (Click-to-Lock & Auto-Enrollment)
        self.video_widget.frame_clicked.connect(self._on_video_frame_clicked)
        self._auto_enroll_success.connect(self._on_auto_enroll_success)
        self._auto_enroll_failed.connect(self._on_auto_enroll_failed)

    @Slot(int, int, str)
    def _on_auto_enroll_success(self, target_tid: int, assigned_id: int, person_name: str) -> None:
        self.runner.lock_target(target_tid, assigned_id)
        self.status_bar.showMessage(
            f"🎯 Đã lấy mẫu & khóa mục tiêu: {person_name} (ID: {assigned_id}) - Track #{target_tid}",
            5000,
        )

    @Slot(int, str)
    def _on_auto_enroll_failed(self, target_tid: int, message: str) -> None:
        self.status_bar.showMessage(message, 4000)

    def _on_frame_processed_from_worker(self, rendered_frame: np.ndarray, results: list, fps: float, latency: float) -> None:
        """Called from worker thread — emits signal only if UI is ready.
        If UI is still rendering previous frame, drop this frame for display to prevent Qt event queue lag.
        """
        if not self._ui_busy.is_set():
            self._ui_busy.set()
            self._frame_ready.emit(rendered_frame, results, fps, latency)

    @Slot(np.ndarray, list, float, float)
    def _handle_frame_on_main_thread(self, rendered_frame: np.ndarray, results: list, fps: float, latency: float) -> None:
        """Runs on main/GUI thread — safe to update all UI widgets."""
        try:
            # Update video display with tracking results and raw frame for live avatar extraction
            self.video_widget.update_frame(
                rendered_frame,
                results=results,
                raw_frame=self.runner._latest_raw_frame,
            )

            # Update enroll dialog if active
            if self._enroll_dialog is not None and self._enroll_dialog.isVisible():
                raw_frame = self.runner._latest_raw_frame
                if raw_frame is not None:
                    self._enroll_dialog.set_live_frame(raw_frame)

            # Queue DB events (non-blocking, offloaded to background writer)
            if self._current_session_id is not None:
                for res in results:
                    prev_id = self._active_track_identities.get(res.track_id)
                    if prev_id != res.person_id and res.person_id is not None:
                        self._active_track_identities[res.track_id] = res.person_id
                        crop_img = None
                        if res.face_bbox is not None and self.runner._latest_raw_frame is not None:
                            x1, y1, x2, y2 = map(int, res.face_bbox)
                            h_f, w_f = self.runner._latest_raw_frame.shape[:2]
                            crop_img = self.runner._latest_raw_frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)].copy()

                        try:
                            self._event_queue.put_nowait({
                                "session_id": self._current_session_id,
                                "track_no": res.track_id,
                                "person_id": res.person_id,
                                "similarity": res.similarity,
                                "quality": res.quality,
                                "is_unknown": res.is_unknown,
                                "bbox": f"{int(res.raw_bbox[0])},{int(res.raw_bbox[1])},{int(res.raw_bbox[2])},{int(res.raw_bbox[3])}",
                                "snapshot_crop": crop_img,
                                "snapshots_dir": self.cfg.storage.snapshots_dir,
                            })
                        except queue.Full:
                            pass  # Drop event if queue is full — non-critical
        finally:
            self._ui_busy.clear()

    def _on_template_learned(self, person_id: int, emb: np.ndarray, quality: float, crop: Optional[np.ndarray]) -> None:
        """Enqueues newly learned diverse face template to background DB writer."""
        try:
            self._event_queue.put_nowait({
                "action": "learn_template",
                "person_id": person_id,
                "embedding": emb,
                "quality": quality,
                "crop": crop,
            })
        except queue.Full:
            pass

    def _db_writer_loop(self) -> None:
        """Background thread consuming event queue and writing to DB."""
        while not self._db_writer_stop.is_set():
            try:
                event_data = self._event_queue.get(timeout=0.5)
                if event_data.get("action") == "learn_template":
                    self.repo.add_template_to_person(
                        person_id=event_data["person_id"],
                        embedding=event_data["embedding"],
                        quality=event_data["quality"],
                        crop_img=event_data.get("crop"),
                        recognizer=None,
                        snapshots_dir=self.cfg.storage.snapshots_dir,
                    )
                    try:
                        self.runner.pipeline.recognizer.save()
                    except Exception as save_err:
                        print(f"[MainWindow] Background FAISS save warning: {save_err}")
                else:
                    self.repo.record_track_event(**event_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MainWindow] DB writer error: {e}")

    def start_pipeline(self) -> None:
        source_data = self.sources_dock.get_current_source()
        if not source_data:
            QMessageBox.warning(self, "Chưa chọn nguồn", "Vui lòng chọn một nguồn video từ dock Nguồn!")
            return

        try:
            source = open_source(source_data["type"], source_data["uri"])
            self.runner.set_source(source)
            self.runner.start()

            self._run_start_time = time.time()
            self._current_session_id = self.repo.start_session(config_snapshot=str(self.cfg.to_dict()))
            self._active_track_identities.clear()

            self.controls_dock.set_running_state(True)
            self.lbl_status.setText("● ĐANG CHẠY")
            self.lbl_status.setStyleSheet("color: #2ea043; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi khởi động", f"Không thể mở nguồn video: {e}")

    def stop_pipeline(self) -> None:
        self.runner.stop(release_source=True)
        if self._current_session_id is not None:
            self.repo.end_session(self._current_session_id)
            self._current_session_id = None

        self._run_start_time = None
        self._rec_start_time = None
        self.controls_dock.set_running_state(False)
        self.controls_dock.set_recording_state(False)
        self.lbl_status.setText("⏹ ĐÃ DỪNG")
        self.lbl_status.setStyleSheet("color: #8f93a2; font-weight: bold;")
        self.video_widget.clear_frame()

    def toggle_recording(self) -> None:
        if not self.runner.is_running:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng khởi động pipeline trước khi ghi hình!")
            return

        if not self.runner.is_recording:
            path = self.runner.start_recording()
            self._rec_start_time = time.time()
            self.controls_dock.set_recording_state(True)
            self.lbl_rec_timer.setStyleSheet("color: #d9383a; font-weight: bold; margin-right: 8px;")
        else:
            path = self.runner.stop_recording()
            self._rec_start_time = None
            self.controls_dock.set_recording_state(False)
            self.lbl_rec_timer.setStyleSheet("color: #777b8f; font-weight: bold; margin-right: 8px;")
            if path:
                QMessageBox.information(self, "Ghi hình hoàn tất", f"Video đã được lưu tại:\n{path}")

    def _on_source_selected(self, source_type: str, uri: str) -> None:
        if self.runner.is_running:
            try:
                new_src = open_source(source_type, uri)
                self.runner.set_source(new_src)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi chuyển nguồn", f"Không thể kết nối đến {uri}: {e}")

    def _save_current_config(self) -> None:
        """Persists current runtime configuration to disk."""
        try:
            os.makedirs(os.path.dirname(self.cfg_path), exist_ok=True)
            self.cfg.save_json(self.cfg_path)
        except Exception as e:
            print(f"[MainWindow] Error saving config to {self.cfg_path}: {e}")

    def _on_mode_changed(self, mode: SystemMode) -> None:
        self.cfg.apply_preset(mode)
        self.overlay_dock.apply_config(self.cfg.overlay)
        self.tuning_dock.apply_config(self.cfg)
        self.runner.update_config(self.cfg)
        self._save_current_config()

    def _on_tuning_changed(self, params: dict) -> None:
        if "match_thr" in params:
            self.cfg.recognition.match_thr = params["match_thr"]
        if "min_quality" in params:
            self.cfg.quality.min_quality = params["min_quality"]
        if "min_cutoff" in params:
            self.cfg.smoothing.min_cutoff = params["min_cutoff"]
        if "beta" in params:
            self.cfg.smoothing.beta = params["beta"]
        if "max_yaw_ratio" in params:
            self.cfg.landmark_filter.max_yaw_ratio = params["max_yaw_ratio"]
        self.runner.update_config(self.cfg)
        self._save_current_config()

    def _on_overlay_changed(self, flags: dict) -> None:
        for k, v in flags.items():
            if hasattr(self.cfg.overlay, k):
                setattr(self.cfg.overlay, k, v)
        self.runner.update_config(self.cfg)
        self._save_current_config()

    def _open_enroll_dialog(self) -> None:
        self._enroll_dialog = EnrollDialog(
            repo=self.repo,
            recognizer=self.runner.pipeline.recognizer,
            face_det=self.runner.pipeline.face_det,
            quality_est=self.runner.pipeline.quality_estimator,
            cfg=self.cfg,
            parent=self,
            inference_lock=self.runner.pipeline._inference_lock,
        )
        self._enroll_dialog.exec()
        self._enroll_dialog = None

    def _open_settings_dialog(self) -> None:
        dlg = SettingsDialog(
            cfg=self.cfg,
            repo=self.repo,
            recognizer=self.runner.pipeline.recognizer,
            parent=self,
        )
        dlg.config_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, new_cfg: AppConfig) -> None:
        self.cfg = new_cfg
        self.runner.update_config(new_cfg)
        self.tuning_dock.apply_config(new_cfg)
        self.overlay_dock.apply_config(new_cfg.overlay)
        self._save_current_config()

    def _open_events_dialog(self) -> None:
        dlg = EventsDialog(repo=self.repo, parent=self)
        dlg.exec()

    def _update_status_bar(self) -> None:
        now = time.time()
        # RUN Timer
        if self._run_start_time is not None:
            el = int(now - self._run_start_time)
            hh, rem = divmod(el, 3600)
            mm, ss = divmod(rem, 60)
            self.lbl_run_timer.setText(f"RUN {hh:02d}:{mm:02d}:{ss:02d}")
        else:
            self.lbl_run_timer.setText("RUN 00:00:00")

        # REC Timer
        if self._rec_start_time is not None:
            el = int(now - self._rec_start_time)
            hh, rem = divmod(el, 3600)
            mm, ss = divmod(rem, 60)
            self.lbl_rec_timer.setText(f"REC {hh:02d}:{mm:02d}:{ss:02d}")
        else:
            self.lbl_rec_timer.setText("REC 00:00:00")

        # FPS, Hardware Accelerator & Latency
        self.lbl_fps.setText(f"FPS {self.runner.fps:.1f}")
        cpu_pct = psutil.cpu_percent()
        providers = getattr(self.runner.pipeline.person_det, "providers", []) or []
        from core.config import get_gpu_device_name
        dev_id = getattr(self.cfg.hardware, "device_id", None)
        gpu_name = get_gpu_device_name(dev_id)
        if any("CUDA" in str(p) for p in providers):
            hw_label = f"GPU: CUDA ({gpu_name})"
        elif any("Dml" in str(p) for p in providers):
            hw_label = f"GPU: DirectML ({gpu_name})"
        elif any("CoreML" in str(p) for p in providers):
            hw_label = f"GPU: CoreML ({gpu_name})"
        else:
            hw_label = "CPU"
        self.lbl_perf.setText(f"CPU {cpu_pct:.0f}% | {hw_label} | {self.runner.latency_ms:.0f}ms")
        self.lbl_tracks.setText(f"Tracks: {len(self.runner._latest_results)}")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "AI Face Studio",
            "<h3>Hệ thống nhận diện khuôn mặt và bám bắt đối tượng</h3>"
            "<p><b>Kiến trúc:</b> YOLOv11n → ByteTrack → RetinaFace → Filter 5-landmark → eDifFIQA-T → EdgeFace → FAISS + SQLite</p>"
            "<p><b>Giao diện:</b> PySide6 OBS Studio Layout</p>"
        )

    def _restore_dock_state(self) -> None:
        settings = QSettings("FaceStudio", "FaceApp")
        geo = settings.value("geometry")
        state = settings.value("windowState")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)

    def _get_next_sequential_person_id(self) -> int:
        """Finds the lowest available positive integer for automatic person ID assignment (1, 2, ... N+)."""
        persons = self.repo.list_persons(include_deleted=False)
        used_numbers = set()
        for p in persons:
            code = str(p.get("code", "")).strip()
            if code.isdigit():
                used_numbers.add(int(code))
            name = str(p.get("full_name", "")).strip()
            if name.lower().startswith("person "):
                parts = name.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    used_numbers.add(int(parts[-1]))
        next_id = 1
        while next_id in used_numbers:
            next_id += 1
        return next_id

    def _extract_immediate_face_sample(
        self, raw_frame: np.ndarray, res
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Extracts aligned face crop, embedding, and quality score directly from raw frame with inference lock."""
        pipe = self.runner.pipeline
        with pipe._inference_lock:
            landmarks = res.landmarks
            if landmarks is None:
                if pipe.cfg.tracking.mode == TrackingMode.FACE:
                    # In Face tracking mode, res.smoothed_bbox is the face bbox
                    h_img, w_img = raw_frame.shape[:2]
                    fx1, fy1, fx2, fy2 = res.smoothed_bbox
                    margin = max(10, int(0.20 * max(fx2 - fx1, fy2 - fy1)))
                    cx1, cy1 = max(0, int(fx1 - margin)), max(0, int(fy1 - margin))
                    cx2, cy2 = min(w_img, int(fx2 + margin)), min(h_img, int(fy2 + margin))
                    sub_crop = raw_frame[cy1:cy2, cx1:cx2]
                    if sub_crop.size > 0:
                        sub_faces = pipe.face_det.detect_in_frame(sub_crop)
                        if sub_faces:
                            best_f = max(sub_faces, key=lambda f: f.conf)
                            if best_f.landmarks is not None:
                                landmarks = best_f.landmarks.copy()
                                landmarks[:, 0] += cx1
                                landmarks[:, 1] += cy1
                else:
                    face_res = pipe.face_det.detect_in_person_crop(raw_frame, res.smoothed_bbox)
                    if face_res is not None and face_res.landmarks is not None:
                        landmarks = face_res.landmarks

            # Fallback: if still None, detect across full frame and match by spatial center
            if landmarks is None:
                full_faces = pipe.face_det.detect_in_frame(raw_frame)
                b = res.smoothed_bbox
                for f in full_faces:
                    if f.landmarks is not None:
                        fcx = (f.bbox[0] + f.bbox[2]) / 2.0
                        fcy = (f.bbox[1] + f.bbox[3]) / 2.0
                        if b[0] <= fcx <= b[2] and b[1] <= fcy <= b[3]:
                            landmarks = f.landmarks
                            break

            if landmarks is None:
                return None, None, 0.0

            try:
                aligned, _ = pipe.recognizer.align(raw_frame, landmarks)
                emb = pipe.recognizer.extract_embedding(aligned)
                q_score = pipe.quality_estimator.score(raw_frame, landmarks)
                return emb, aligned, float(q_score)
            except Exception as e:
                print(f"[MainWindow] Immediate sample extraction failed: {e}")
                return None, None, 0.0


    @Slot(float, float)
    def _on_video_frame_clicked(self, frame_x: float, frame_y: float) -> None:
        """Handles user clicking directly on a person in the video viewport.
        
        Behavior:
        1. Find clicked person/track.
        2. If already locked, toggle unlock.
        3. If known person, lock directly.
        4. If unknown person, sample face immediately, auto-assign ID 1 (then 2, 3... N+),
           enroll to DB & FAISS, and lock onto target.
        """
        if not self.runner.is_running:
            self.status_bar.showMessage("Vui lòng khởi động pipeline trước khi chọn mục tiêu.", 3000)
            return

        latest_frame, results, fps, latency = self.runner.get_latest()
        raw_frame = self.runner._latest_raw_frame
        if not results or raw_frame is None:
            return

        # Find candidate tracks containing (frame_x, frame_y)
        clicked_candidates = []
        for r in results:
            b = r.smoothed_bbox
            if b[0] <= frame_x <= b[2] and b[1] <= frame_y <= b[3]:
                area = (b[2] - b[0]) * (b[3] - b[1])
                clicked_candidates.append((area, r))

        # If clicked outside any person bounding box
        if not clicked_candidates:
            curr_locked_tid, curr_locked_pid = self.runner.locked_target
            if curr_locked_tid is not None or curr_locked_pid is not None:
                self.runner.unlock_target()
                self.status_bar.showMessage("Đã hủy khóa mục tiêu (Unlocked)", 3000)
            return

        # Prioritize innermost/smallest box if multiple tracks overlap
        clicked_candidates.sort(key=lambda item: item[0])
        target_res = clicked_candidates[0][1]
        target_tid = target_res.track_id

        # Toggle unlock if clicking on the currently locked target
        curr_locked_tid, curr_locked_pid = self.runner.locked_target
        if (curr_locked_tid is not None and curr_locked_tid == target_tid) or (
            curr_locked_pid is not None and target_res.person_id == curr_locked_pid
        ):
            self.runner.unlock_target()
            self.status_bar.showMessage(f"Đã hủy khóa mục tiêu: {target_res.person_name} (Track #{target_tid})", 3000)
            return

        # Case A: Person is already enrolled and confirmed
        is_already_known = (
            target_res.person_id is not None
            and not target_res.is_unknown
            and (curr_locked_pid is None or target_res.person_id != curr_locked_pid)
        )

        if is_already_known:
            self.runner.lock_target(target_tid, target_res.person_id)
            self.status_bar.showMessage(f"🎯 Đã khóa mục tiêu: {target_res.person_name} (Track #{target_tid})", 4000)
            return

        # Case B: Sample face and auto-assign next ID (ID 1, then N+)
        # Lock target track immediately for UI responsiveness
        self.runner.lock_target(target_tid)
        self.status_bar.showMessage(f"🎯 Đang phân tích mẫu khuôn mặt cho Track #{target_tid}...", 2000)
        worker = EnrollWorker(self, raw_frame.copy(), target_res, target_tid)
        QThreadPool.globalInstance().start(worker)

    def _async_sample_and_enroll(self, raw_frame: np.ndarray, target_res, target_tid: int) -> None:
        """Executes face sampling and enrollment on background worker thread."""
        state = self.runner.pipeline.tracker.states.get(target_tid)
        best_emb = state.best_embedding if state else None
        best_crop = state.best_face_crop if state else None
        best_quality = state.best_quality if state else 0.0

        # If live state doesn't have an embedding yet, extract from raw_frame (in worker thread)
        if best_emb is None or best_crop is None:
            best_emb, best_crop, best_quality = self._extract_immediate_face_sample(raw_frame, target_res)

        if best_emb is None or best_crop is None:
            self._auto_enroll_failed.emit(
                target_tid,
                f"⚠️ Chưa thể lấy mẫu Track #{target_tid}: Không phát hiện rõ khuôn mặt. Vui lòng thử lại khi đối diện camera!",
            )
            return

        # Deduplication check: does this face already match an enrolled person in the DB?
        match_meta, match_sim = self.runner.pipeline.recognizer.search(
            best_emb, threshold=self.cfg.recognition.match_thr
        )
        if match_meta is not None and match_meta.get("person_id") is not None:
            matched_pid = match_meta["person_id"]
            matched_name = match_meta.get("full_name", match_meta.get("name", f"Person {matched_pid}"))

            # Person already exists in DB! Re-link to existing identity instead of duplicating
            if state is not None:
                state.person_id = matched_pid
                state.person_name = matched_name
                state.is_unknown = False
                state.similarity = match_sim
                state.votes.clear()
                state.vote(matched_pid, matched_name, match_sim)
                state.gallery.set_anchor(best_emb, best_quality, best_crop)

            self.runner.lock_target(target_tid, matched_pid)
            self._auto_enroll_success.emit(target_tid, matched_pid, matched_name)
            return

        # Determine next sequential ID: 1, 2, ... N+
        assigned_id = self._get_next_sequential_person_id()
        person_code = str(assigned_id)
        person_name = f"Person {assigned_id}"

        try:
            person = self.repo.enroll_face(
                code=person_code,
                full_name=person_name,
                group_name="Auto Locked",
                templates=[(best_emb, best_quality, best_crop)],
                recognizer=self.runner.pipeline.recognizer,
                snapshots_dir=self.cfg.storage.snapshots_dir,
            )

            # Update live track state immediately
            if state is not None:
                state.person_id = person.id
                state.person_name = person_name
                state.is_unknown = False
                state.similarity = 1.0
                state.votes.clear()
                state.vote(person.id, person_name, 1.0)
                state.gallery.set_anchor(best_emb, best_quality, best_crop)

            self._auto_enroll_success.emit(target_tid, person.id, person_name)
        except Exception as e:
            print(f"[MainWindow] Async enroll error: {e}")
            self._auto_enroll_failed.emit(target_tid, f"Lỗi lấy mẫu: {e}")

    def _confirm_and_reset_database(self) -> None:
        """Prompts user for confirmation and completely resets database, FAISS index, and snapshots."""
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa cơ sở dữ liệu",
            "Bạn có chắc chắn muốn xóa toàn bộ danh sách người đã lưu, các vector khuôn mặt FAISS và lịch sử nhận diện?\n\n"
            "⚠️ Thao tác này KHÔNG THỂ HOÀN TÁC!\n"
            "Sau khi xóa, lượt lấy mẫu tiếp theo sẽ bắt đầu lại từ ID 1.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.runner.unlock_target()
            try:
                self.repo.reset_database(
                    recognizer=self.runner.pipeline.recognizer,
                    clear_snapshots=True,
                    snapshots_dir=self.cfg.storage.snapshots_dir,
                )

                # Reset FAISS active person set so soft-delete filter doesn't block new IDs
                self.runner.pipeline.recognizer.set_active_persons(None)

                # Reset ALL in-memory track states (person_id, votes, gallery, embedding, face state)
                for state in list(self.runner.pipeline.tracker.states.values()):
                    state.person_id = None
                    state.person_name = "Unknown"
                    state.is_unknown = True
                    state.votes.clear()
                    state.similarity = 0.0
                    state.best_embedding = None
                    state.best_face_crop = None
                    state.best_quality = 0.0
                    state.last_face_box = None
                    state.relative_face_box = None
                    state.last_recognition_time = 0.0
                    # Reset face gallery (learning templates)
                    from core.tracker import FaceGallery
                    state.gallery = FaceGallery(target_quota=5)

                # Clear pipeline-level lock state
                self.runner.pipeline.locked_track_id = None
                self.runner.pipeline.locked_person_id = None

                # Clear UI-level caches
                if hasattr(self, "_active_track_identities"):
                    self._active_track_identities.clear()

                # Clear video widget face thumbnails
                if hasattr(self.video_widget, "_face_thumbnails"):
                    self.video_widget._face_thumbnails.clear()

                self.status_bar.showMessage("✅ Đã xóa toàn bộ cơ sở dữ liệu! Sẵn sàng lấy mẫu mới từ ID 1.", 6000)
                QMessageBox.information(
                    self,
                    "Đã xóa dữ liệu",
                    "✅ Cơ sở dữ liệu khuôn mặt và chỉ mục vector FAISS đã được dọn sạch hoàn toàn.\n\n"
                    "• Tất cả danh tính đã nhận diện → Unknown\n"
                    "• Lịch sử nhận diện đã xóa\n"
                    "• Lượt nhấn chọn mục tiêu tiếp theo sẽ tự động được gán là Person 1 (ID 1).",
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi khi xóa", f"Không thể xóa cơ sở dữ liệu: {e}")

    def closeEvent(self, event) -> None:
        # Save state
        settings = QSettings("FaceStudio", "FaceApp")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        self._save_current_config()

        self.stop_pipeline()
        # Stop DB writer thread
        self._db_writer_stop.set()
        self._db_writer_thread.join(timeout=2.0)
        event.accept()
