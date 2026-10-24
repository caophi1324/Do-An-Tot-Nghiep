"""PipelineRunner executing video capture and pipeline processing in a background worker thread."""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from .config import AppConfig
from .pipeline import Pipeline, render_overlay
from .sources import VideoSource, open_source
from .tracker import TrackResult


class PipelineRunner:
    """Manages the background execution loop, FPS calculation, and frame broadcasting."""

    def __init__(
        self,
        config: AppConfig,
        source: Optional[VideoSource] = None,
        on_frame_processed: Optional[Callable[[np.ndarray, List[TrackResult], float, float], None]] = None,
        on_template_learned: Optional[Callable[[int, np.ndarray, float, Optional[np.ndarray]], None]] = None,
    ) -> None:
        self.config = config
        self.source = source
        self.pipeline = Pipeline(config, on_template_learned=on_template_learned)
        self.on_frame_processed = on_frame_processed

        self._running = False
        self._paused = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._source_lock = threading.RLock()
        self._writer_lock = threading.Lock()

        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_rendered_frame: Optional[np.ndarray] = None
        self._latest_results: List[TrackResult] = []
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_subscribers: int = 0
        self._new_frame_cond = threading.Condition(self._lock)

        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frame_count: int = 0

        # Recording
        self._is_recording = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_recording(self) -> bool:
        with self._writer_lock:
            return self._is_recording

    @property
    def _has_jpeg_subscribers(self) -> bool:
        with self._lock:
            return self._jpeg_subscribers > 0

    @_has_jpeg_subscribers.setter
    def _has_jpeg_subscribers(self, val: bool) -> None:
        with self._lock:
            self._jpeg_subscribers = 1 if val else 0

    def subscribe_jpeg(self) -> None:
        """Increments active MJPEG subscriber count."""
        with self._lock:
            self._jpeg_subscribers += 1

    def unsubscribe_jpeg(self) -> None:
        """Decrements active MJPEG subscriber count."""
        with self._lock:
            self._jpeg_subscribers = max(0, self._jpeg_subscribers - 1)

    @property
    def source_connected(self) -> bool:
        with self._source_lock:
            return self.source.is_connected() if self.source is not None else False

    def set_source(self, source: VideoSource) -> None:
        """Swaps or updates the video source safely without deadlocks or thread leaks."""
        was_running = self._running
        if was_running:
            self.stop(release_source=False)

        with self._source_lock:
            if self.source is not None:
                try:
                    self.source.stop()
                except Exception as e:
                    print(f"[PipelineRunner] Error stopping previous source: {e}")
            self.source = source

        if was_running:
            self.start()

    def update_config(self, new_config: AppConfig) -> None:
        """Applies updated configuration to pipeline and overlay."""
        with self._lock:
            self.config = new_config
        self.pipeline.update_config(new_config)

    def lock_target(self, track_id: int, person_id: Optional[int] = None) -> None:
        """Locks onto a specific target by track_id and/or person_id without holding runner lock."""
        self.pipeline.lock_target(track_id, person_id)

    def unlock_target(self) -> None:
        """Clears the locked target without holding runner lock."""
        self.pipeline.unlock_target()

    @property
    def locked_target(self) -> Tuple[Optional[int], Optional[int]]:
        return self.pipeline.locked_track_id, self.pipeline.locked_person_id

    def start(self) -> None:
        """Starts the pipeline processing loop (ensures strictly single worker thread)."""
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                self._running = True
                return
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self, release_source: bool = True) -> None:
        """Stops the processing loop and recording."""
        with self._thread_lock:
            self._running = False
            self._stop_event.set()
            if self._is_recording:
                self.stop_recording()
            if self._thread is not None and self._thread.is_alive():
                if threading.current_thread() != self._thread:
                    self._thread.join(timeout=2.0)
            self._thread = None

        if release_source:
            with self._source_lock:
                if self.source is not None:
                    try:
                        self.source.stop()
                    except Exception as e:
                        print(f"[PipelineRunner] Error releasing source: {e}")

    def start_recording(self, output_dir: Optional[str] = None) -> str:
        """Starts saving processed video frames to an MP4 file."""
        with self._writer_lock:
            if self._is_recording:
                return self._recording_path or ""

            rec_dir = output_dir or self.config.storage.recordings_dir
            os.makedirs(rec_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self._recording_path = os.path.join(rec_dir, f"recording_{timestamp_str}.mp4")
            self._is_recording = True
            return self._recording_path

    def stop_recording(self) -> Optional[str]:
        """Stops video recording and finalizes MP4 file with lock protection."""
        with self._writer_lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            writer = self._video_writer
            self._video_writer = None
            path = self._recording_path
            self._recording_path = None
            if writer is not None:
                try:
                    writer.release()
                except Exception as e:
                    print(f"[PipelineRunner] Error releasing VideoWriter: {e}")
            return path

    def _run_loop(self) -> None:
        prev_time = time.time()
        prev_frame_ts = -1.0
        fps_smoothing = 0.9

        while not self._stop_event.is_set():
            with self._source_lock:
                src = self.source
            if src is None or not src.is_connected():
                time.sleep(0.05)
                continue

            ts, frame = src.read()
            # Ignore stale or identical timestamp frame
            if frame is None or ts == prev_frame_ts:
                time.sleep(0.005)
                continue
            prev_frame_ts = ts

            t0 = time.time()
            # Run inference pipeline
            results = self.pipeline.process(frame, timestamp=ts)
            t1 = time.time()

            latency = (t1 - t0) * 1000.0  # ms
            instant_fps = 1.0 / max(1e-4, (t1 - prev_time))
            self.fps = fps_smoothing * self.fps + (1.0 - fps_smoothing) * instant_fps
            self.latency_ms = latency
            prev_time = t1
            self.frame_count += 1

            # Render overlay
            rendered = render_overlay(
                frame=frame,
                results=results,
                cfg=self.config.overlay,
                fps=self.fps,
                latency_ms=self.latency_ms,
            )

            # Handle recording safely
            try:
                with self._writer_lock:
                    if self._is_recording and self._recording_path is not None:
                        h, w = rendered.shape[:2]
                        if self._video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            rec_fps = max(5.0, min(60.0, self.fps if self.fps > 0.0 else 25.0))
                            self._video_writer = cv2.VideoWriter(self._recording_path, fourcc, rec_fps, (w, h))
                        if self._video_writer is not None:
                            self._video_writer.write(rendered)
            except Exception as e:
                print(f"[PipelineRunner] VideoWriter write error: {e}")

            # Encode JPEG only if there are active stream subscribers
            jpeg_bytes = None
            if self._has_jpeg_subscribers:
                ok, buf = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = buf.tobytes() if ok else None

            with self._lock:
                self._latest_raw_frame = frame
                self._latest_rendered_frame = rendered
                self._latest_results = results
                self._latest_jpeg = jpeg_bytes
                self._new_frame_cond.notify_all()

            # Trigger optional callback
            if self.on_frame_processed is not None:
                try:
                    self.on_frame_processed(rendered, results, self.fps, self.latency_ms)
                except Exception as e:
                    print(f"[PipelineRunner] Callback error: {e}")

        # Cleanup on exit
        with self._writer_lock:
            if self._video_writer is not None:
                try:
                    self._video_writer.release()
                except Exception:
                    pass
                self._video_writer = None

    def get_latest(self) -> Tuple[Optional[np.ndarray], List[TrackResult], float, float]:
        with self._lock:
            fps_snapshot = self.fps
            latency_snapshot = self.latency_ms
            return (
                self._latest_rendered_frame,
                list(self._latest_results),
                fps_snapshot,
                latency_snapshot,
            )

    def wait_next_jpeg(self, timeout: float = 1.0) -> Optional[bytes]:
        """Waits for the next processed JPEG frame (used for FastAPI MJPEG stream)."""
        with self._lock:
            if self._new_frame_cond.wait(timeout=timeout):
                if self._latest_jpeg is None and self._latest_rendered_frame is not None:
                    ok, buf = cv2.imencode(".jpg", self._latest_rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._latest_jpeg = buf.tobytes() if ok else None
                return self._latest_jpeg
            return None
