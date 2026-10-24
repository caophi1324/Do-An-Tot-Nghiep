"""Tuning Dock widget for real-time parameter adjustment with sliders."""
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QDockWidget,
)
from core.config import AppConfig


class TuningDock(QDockWidget):
    """OBS 'Audio Mixer' analog: Real-time system tuning sliders."""

    tuning_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, cfg: Optional[AppConfig] = None) -> None:
        super().__init__("Tinh chỉnh (Tuning)", parent)
        self.setObjectName("TuningDock")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        # Debounce timer to prevent signal spam
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._emit_pending_tuning)
        self._pending_tuning: dict = {}

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1. Match Threshold Slider
        self.match_val_label = QLabel("0.50")
        self.match_slider = self._create_slider_row(
            layout,
            "Ngưỡng nhận diện:",
            self.match_val_label,
            10, 90, 50,
            self._on_match_thr_changed,
        )

        # 2. Quality Score Slider
        self.quality_val_label = QLabel("0.30")
        self.quality_slider = self._create_slider_row(
            layout,
            "Quality tối thiểu:",
            self.quality_val_label,
            0, 100, 30,
            self._on_quality_changed,
        )

        # 3. Smoothing Cutoff Slider
        self.cutoff_val_label = QLabel("1.0")
        self.cutoff_slider = self._create_slider_row(
            layout,
            "Độ mượt (Cutoff):",
            self.cutoff_val_label,
            1, 50, 10,
            self._on_cutoff_changed,
        )

        # 4. Smoothing Beta Slider
        self.beta_val_label = QLabel("0.010")
        self.beta_slider = self._create_slider_row(
            layout,
            "Độ nhạy (Beta):",
            self.beta_val_label,
            1, 100, 10,
            self._on_beta_changed,
        )

        # 5. Max Yaw Angle
        self.yaw_val_label = QLabel("0.45")
        self.yaw_slider = self._create_slider_row(
            layout,
            "Góc quay ngang (Yaw):",
            self.yaw_val_label,
            10, 80, 45,
            self._on_yaw_changed,
        )

        # Reset button
        self.reset_btn = QPushButton("Khôi phục mặc định")
        layout.addWidget(self.reset_btn)
        self.reset_btn.clicked.connect(self.reset_defaults)

        self.setWidget(container)
        self.apply_config(cfg if cfg is not None else AppConfig())

    def _create_slider_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        value_label: QLabel,
        min_val: int,
        max_val: int,
        default_val: int,
        callback,
    ) -> QSlider:
        row_layout = QVBoxLayout()
        row_layout.setSpacing(2)

        header = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #b0b4c3; font-size: 8.5pt;")
        value_label.setStyleSheet("color: #2b78e4; font-weight: bold; font-size: 8.5pt;")
        header.addWidget(lbl)
        header.addStretch()
        header.addWidget(value_label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        # Update label in real-time and queue debounced change
        slider.valueChanged.connect(callback)
        # Immediate emit when mouse releases
        slider.sliderReleased.connect(self._emit_pending_tuning)

        row_layout.addLayout(header)
        row_layout.addWidget(slider)
        parent_layout.addLayout(row_layout)
        return slider

    def _queue_tuning_change(self, key: str, val: float) -> None:
        self._pending_tuning[key] = val
        self._debounce_timer.start()

    def _emit_pending_tuning(self) -> None:
        self._debounce_timer.stop()
        if self._pending_tuning:
            to_emit = dict(self._pending_tuning)
            self._pending_tuning.clear()
            self.tuning_changed.emit(to_emit)

    def _on_match_thr_changed(self, val: int) -> None:
        thr = val / 100.0
        self.match_val_label.setText(f"{thr:.2f}")
        self._queue_tuning_change("match_thr", thr)

    def _on_quality_changed(self, val: int) -> None:
        q = val / 100.0
        self.quality_val_label.setText(f"{q:.2f}")
        self._queue_tuning_change("min_quality", q)

    def _on_cutoff_changed(self, val: int) -> None:
        cutoff = val / 10.0
        self.cutoff_val_label.setText(f"{cutoff:.1f}")
        self._queue_tuning_change("min_cutoff", cutoff)

    def _on_beta_changed(self, val: int) -> None:
        beta = val / 1000.0
        self.beta_val_label.setText(f"{beta:.3f}")
        self._queue_tuning_change("beta", beta)

    def _on_yaw_changed(self, val: int) -> None:
        yaw = val / 100.0
        self.yaw_val_label.setText(f"{yaw:.2f}")
        self._queue_tuning_change("max_yaw_ratio", yaw)

    def apply_config(self, cfg: AppConfig) -> None:
        """Applies configuration parameters to sliders and labels without emitting tuning_changed."""
        self._debounce_timer.stop()
        self._pending_tuning.clear()

        m_thr = cfg.recognition.match_thr
        q_min = cfg.quality.min_quality
        cutoff = cfg.smoothing.min_cutoff
        beta = cfg.smoothing.beta
        yaw = cfg.landmark_filter.max_yaw_ratio

        self.match_slider.blockSignals(True)
        self.match_slider.setValue(int(round(m_thr * 100)))
        self.match_slider.blockSignals(False)
        self.match_val_label.setText(f"{m_thr:.2f}")

        self.quality_slider.blockSignals(True)
        self.quality_slider.setValue(int(round(q_min * 100)))
        self.quality_slider.blockSignals(False)
        self.quality_val_label.setText(f"{q_min:.2f}")

        self.cutoff_slider.blockSignals(True)
        self.cutoff_slider.setValue(int(round(cutoff * 10)))
        self.cutoff_slider.blockSignals(False)
        self.cutoff_val_label.setText(f"{cutoff:.1f}")

        self.beta_slider.blockSignals(True)
        self.beta_slider.setValue(int(round(beta * 1000)))
        self.beta_slider.blockSignals(False)
        self.beta_val_label.setText(f"{beta:.3f}")

        self.yaw_slider.blockSignals(True)
        self.yaw_slider.setValue(int(round(yaw * 100)))
        self.yaw_slider.blockSignals(False)
        self.yaw_val_label.setText(f"{yaw:.2f}")

    def reset_defaults(self) -> None:
        """Resets sliders to official AppConfig defaults and notifies listener."""
        self._debounce_timer.stop()
        self._pending_tuning.clear()
        default_cfg = AppConfig()
        self.apply_config(default_cfg)
        self.tuning_changed.emit({
            "match_thr": default_cfg.recognition.match_thr,
            "min_quality": default_cfg.quality.min_quality,
            "min_cutoff": default_cfg.smoothing.min_cutoff,
            "beta": default_cfg.smoothing.beta,
            "max_yaw_ratio": default_cfg.landmark_filter.max_yaw_ratio,
        })

