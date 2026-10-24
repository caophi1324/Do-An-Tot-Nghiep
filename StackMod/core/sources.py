"""Video source capture abstractions for Webcam, DroidCam, RTSP, and Video Files."""
from __future__ import annotations

import abc
import os
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoSource(abc.ABC):
    """Abstract interface for all video input sources."""

    @abc.abstractmethod
    def read(self) -> Tuple[float, Optional[np.ndarray]]:
        """Returns (timestamp, frame). If no frame is available, frame is None."""
        pass

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the source is actively delivering frames."""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops reading and releases resources."""
        pass

    @abc.abstractmethod
    def get_info(self) -> dict:
        """Returns metadata such as resolution, fps, source type."""
        pass


class WebcamSource(VideoSource):
    """Captures from local USB or integrated webcam in a dedicated thread."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.device_index = int(device_index)
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_counter: int = 0
        self._last_read_counter: int = 0
        self._connected = False
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if self.width > 0 and self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._connected = cap.isOpened()
        while not self._stop_event.is_set():
            if not self._connected:
                time.sleep(1.0)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                self._connected = cap.isOpened()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self._connected = False
                cap.release()
                time.sleep(0.5)
                continue

            now = time.time()
            with self._frame_cond:
                self._latest_frame = frame
                self._latest_ts = now
                self._frame_counter += 1
                self._connected = True
                self._frame_cond.notify_all()

        if cap is not None:
            cap.release()
        self._connected = False

    def read(self, timeout: float = 0.5) -> Tuple[float, Optional[np.ndarray]]:
        with self._frame_cond:
            if self._last_read_counter >= self._frame_counter:
                if timeout > 0 and not self._stop_event.is_set():
                    self._frame_cond.wait(timeout=timeout)
            if self._frame_counter > self._last_read_counter and self._latest_frame is not None:
                self._last_read_counter = self._frame_counter
                return self._latest_ts, self._latest_frame.copy()
            return 0.0, None

    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_info(self) -> dict:
        return {
            "type": "webcam",
            "device_index": self.device_index,
            "connected": self._connected,
        }


def normalize_stream_uri(uri: str) -> str:
    """Normalize stream URI for DroidCam, RTSP, and HTTP endpoints."""
    uri = str(uri).strip()
    if not uri:
        return uri
    # If missing protocol, prepend http://
    if not (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("rtsp://")):
        uri = "http://" + uri
    # If DroidCam default port 4747 without path, append /video (MJPEG stream)
    try:
        from urllib.parse import urlparse
        parsed = urlparse(uri)
        if ":4747" in parsed.netloc and (not parsed.path or parsed.path == "/"):
            uri = uri.rstrip("/") + "/video"
    except Exception:
        pass
    return uri


class StreamSource(VideoSource):
    """Captures MJPEG / RTSP / HTTP streams (e.g. DroidCam, IP cameras) with auto-reconnect."""

    def __init__(self, uri: str, reconnect_interval_s: float = 3.0) -> None:
        self.uri = normalize_stream_uri(str(uri))
        self.reconnect_interval_s = float(reconnect_interval_s)

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_counter: int = 0
        self._last_read_counter: int = 0
        self._connected = False
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ]
        while not self._stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG, params)
            except Exception:
                try:
                    cap = cv2.VideoCapture(self.uri)
                except Exception:
                    cap = None

            self._connected = bool(cap is not None and cap.isOpened())
            if self._connected and cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            consecutive_failures = 0
            while not self._stop_event.is_set() and self._connected and cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    continue

                consecutive_failures = 0
                now = time.time()
                with self._frame_cond:
                    self._latest_frame = frame
                    self._latest_ts = now
                    self._frame_counter += 1
                    self._connected = True
                    self._frame_cond.notify_all()

            if cap is not None:
                cap.release()
            self._connected = False
            self._stop_event.wait(self.reconnect_interval_s)

    def read(self, timeout: float = 0.5) -> Tuple[float, Optional[np.ndarray]]:
        with self._frame_cond:
            if self._last_read_counter >= self._frame_counter:
                if timeout > 0 and not self._stop_event.is_set():
                    self._frame_cond.wait(timeout=timeout)
            if self._frame_counter > self._last_read_counter and self._latest_frame is not None:
                self._last_read_counter = self._frame_counter
                return self._latest_ts, self._latest_frame.copy()
            return 0.0, None

    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=6.0)  # > timeout (5s) to allow graceful exit
        if self._thread.is_alive():
            # Thread still stuck — it's a daemon thread, will die with process
            print("[StreamSource] Warning: Worker thread did not exit in time (daemon)")

    def get_info(self) -> dict:
        return {
            "type": "stream",
            "uri": self.uri,
            "connected": self._connected,
        }


class FileSource(VideoSource):
    """Reads a video file with loop and playback rate options."""

    def __init__(self, file_path: str, loop: bool = True, fps: Optional[float] = None) -> None:
        self.file_path = str(file_path)
        self.loop = loop
        self.target_fps = fps

        self.cap = cv2.VideoCapture(self.file_path)
        self._connected = self.cap.isOpened()

        native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = native_fps if native_fps > 0 else 30.0
        if self.target_fps is not None and self.target_fps > 0:
            self.fps = self.target_fps
        self.frame_delay = 1.0 / self.fps

        self._lock = threading.Lock()
        self._next_frame_time = time.perf_counter()  # Absolute timing
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Tuple[float, Optional[np.ndarray]]:
        with self._lock:
            if not self._connected or not self.cap.isOpened():
                return 0.0, None

            # Pace playback using absolute time to prevent drift
            now = time.perf_counter()
            wait_time = self._next_frame_time - now
            if wait_time > 0:
                time.sleep(wait_time)

            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        self._connected = False
                        return 0.0, None
                else:
                    self._connected = False
                    return 0.0, None

            self._next_frame_time += self.frame_delay
            # Reset nếu bị trễ quá nhiều (tránh burst frames)
            if time.perf_counter() - self._next_frame_time > self.frame_delay * 3:
                self._next_frame_time = time.perf_counter()
            return time.time(), frame

    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
            self._connected = False

    def seek(self, frame_index: int) -> None:
        with self._lock:
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def get_info(self) -> dict:
        return {
            "type": "file",
            "file_path": self.file_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "connected": self._connected,
        }


def open_source(
    source_type: str,
    uri: str,
    width: int = 1280,
    height: int = 720,
    loop: bool = True,
) -> VideoSource:
    """Factory creating appropriate VideoSource instance."""
    st = source_type.lower().strip()
    if st == "webcam":
        try:
            dev_idx = int(uri)
        except ValueError:
            dev_idx = 0
        return WebcamSource(device_index=dev_idx, width=width, height=height)
    elif st in ("stream", "droidcam", "rtsp", "http"):
        return StreamSource(uri=uri)
    elif st == "file":
        return FileSource(file_path=uri, loop=loop)
    else:
        raise ValueError(f"Unknown source type: '{source_type}'")
