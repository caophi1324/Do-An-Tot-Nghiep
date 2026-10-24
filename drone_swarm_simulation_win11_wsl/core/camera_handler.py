"""
DRONEEXE Core - Camera Handler
Nhận luồng MJPEG từ ESP32-CAM qua HTTPS và cung cấp frame cho AI pipeline.

Yêu cầu:
    pip install opencv-python requests

Sử dụng:
    cam = CameraHandler("https://192.168.4.2:81/stream")
    cam.on_frame(my_callback)
    await cam.start()
    ...
    await cam.stop()
"""

import asyncio
import time
import threading
from typing import Optional, Callable
import numpy as np

from loguru import logger


class CameraHandler:
    """
    Đọc luồng MJPEG từ ESP32-CAM qua HTTPS (hoặc HTTP).
    Chạy trong background thread riêng để không block asyncio event loop.

    Thiết kế:
    - Background thread: requests.get() blocking-read MJPEG boundary
    - Mỗi frame hoàn chỉnh → decode bằng OpenCV → gọi on_frame callbacks
    - asyncio-safe: callbacks được schedule lên event loop chính
    """

    def __init__(
        self,
        stream_url: str,
        verify_ssl: bool = False,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
    ):
        """
        Args:
            stream_url: URL stream, ví dụ "https://192.168.4.2:81/stream"
                        hoặc "https://192.168.4.2/stream"
            verify_ssl:  False để bỏ qua self-signed cert của ESP32 (mặc định)
            connect_timeout: Timeout kết nối ban đầu (giây)
            read_timeout:    Timeout đọc từng chunk (giây)
        """
        self.stream_url = stream_url
        self._verify_ssl = verify_ssl
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks: fn(frame: np.ndarray) -> None
        self._frame_callbacks: list[Callable[[np.ndarray], None]] = []

        # Stats
        self._frame_count = 0
        self._last_fps_time = 0.0
        self._fps = 0.0
        self.latest_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def on_frame(self, callback: Callable[[np.ndarray], None]):
        """
        Đăng ký callback nhận frame mỗi khi có frame mới.
        Callback chạy trên asyncio event loop chính (thread-safe).

        Args:
            callback: fn(frame: np.ndarray) — frame là BGR image (OpenCV format)
        """
        self._frame_callbacks.append(callback)

    async def start(self) -> bool:
        """Bắt đầu đọc stream trong background thread."""
        if self._running:
            logger.warning("CameraHandler đã đang chạy")
            return True

        try:
            import cv2  # noqa: F401 — kiểm tra có opencv chưa
            import requests  # noqa: F401
        except ImportError as e:
            logger.error(f"Thiếu thư viện: {e}. Chạy: pip install opencv-python requests")
            return False

        self._loop = asyncio.get_event_loop()
        self._running = True
        self._thread = threading.Thread(
            target=self._stream_loop, name="CameraStream", daemon=True
        )
        self._thread.start()
        logger.info(f"Camera stream bắt đầu: {self.stream_url}")
        return True

    async def stop(self):
        """Dừng stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Camera stream đã dừng")

    @property
    def fps(self) -> float:
        """FPS thực tế đo được."""
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    #  Background streaming thread                                         #
    # ------------------------------------------------------------------ #

    def _stream_loop(self):
        """
        Chạy trong thread riêng. Đọc MJPEG multipart stream theo ranh giới boundary.
        MJPEG format:
            --<boundary>\r\n
            Content-Type: image/jpeg\r\n
            Content-Length: <size>\r\n
            \r\n
            <jpeg bytes>
            \r\n
            --<boundary>...
        """
        import requests
        import cv2

        while self._running:
            try:
                logger.debug(f"Đang kết nối stream: {self.stream_url}")
                response = requests.get(
                    self.stream_url,
                    stream=True,
                    verify=self._verify_ssl,
                    timeout=(self._connect_timeout, self._read_timeout),
                )

                if response.status_code != 200:
                    logger.error(f"HTTP {response.status_code} từ camera")
                    time.sleep(2)
                    continue

                # Tìm boundary từ Content-Type header
                # Ví dụ: "multipart/x-mixed-replace; boundary=frame"
                content_type = response.headers.get("Content-Type", "")
                boundary = self._parse_boundary(content_type)
                if not boundary:
                    logger.warning(f"Không tìm thấy boundary trong: {content_type}")
                    boundary = "frame"  # ESP32-CAM mặc định

                boundary_bytes = f"--{boundary}".encode()
                buffer = b""

                for chunk in response.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buffer += chunk

                    # Tìm JPEG bắt đầu (SOI) và kết thúc (EOI)
                    start = buffer.find(b"\xff\xd8")  # JPEG SOI
                    end = buffer.find(b"\xff\xd9")    # JPEG EOI

                    if start != -1 and end != -1 and end > start:
                        jpg_data = buffer[start : end + 2]
                        buffer = buffer[end + 2:]

                        # Decode JPEG → numpy array
                        frame = cv2.imdecode(
                            np.frombuffer(jpg_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )

                        if frame is not None:
                            self._on_new_frame(frame)
                        else:
                            logger.debug("Không decode được frame")

            except requests.exceptions.ConnectionError:
                logger.warning("Camera mất kết nối, thử lại sau 2s...")
                time.sleep(2)
            except requests.exceptions.Timeout:
                logger.warning("Camera timeout, thử lại...")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Lỗi stream: {e}")
                time.sleep(2)

    def _on_new_frame(self, frame: np.ndarray):
        """Xử lý frame mới: cập nhật stats và notify callbacks."""
        self.latest_frame = frame
        self._frame_count += 1

        # Tính FPS mỗi giây
        now = time.monotonic()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / max(now - self._last_fps_time, 0.001)
            self._frame_count = 0
            self._last_fps_time = now

        # Schedule callbacks lên asyncio event loop (thread-safe)
        if self._loop and self._frame_callbacks:
            for cb in self._frame_callbacks:
                self._loop.call_soon_threadsafe(
                    lambda c=cb, f=frame: self._safe_callback(c, f)
                )

    @staticmethod
    def _safe_callback(callback: Callable, frame: np.ndarray):
        """Gọi callback, bắt exception để không crash stream thread."""
        try:
            callback(frame)
        except Exception as e:
            logger.error(f"Lỗi frame callback: {e}")

    @staticmethod
    def _parse_boundary(content_type: str) -> Optional[str]:
        """Parse boundary từ Content-Type header."""
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                return part[len("boundary="):]
        return None


# ------------------------------------------------------------------ #
#  Tiện ích: Hiển thị stream bằng OpenCV window (cho debug/demo)     #
# ------------------------------------------------------------------ #

class CameraViewer:
    """
    Wrapper tiện lợi: hiển thị CameraHandler trong OpenCV window.
    Không dùng trong production, chỉ dùng để test/debug.

    Cách dùng:
        viewer = CameraViewer(cam, window_title="DroneEXE FPV")
        await viewer.show()   # block cho đến khi nhấn 'q'
    """

    def __init__(self, handler: "CameraHandler", window_title: str = "DRONEEXE Camera"):
        self._handler = handler
        self._title = window_title
        self._latest: Optional[np.ndarray] = None
        self._stopped = False

    async def show(self, overlay_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Hiển thị stream. Nhấn 'q' để thoát.

        Args:
            overlay_fn: Hàm tùy chọn nhận frame gốc và trả về frame đã xử lý
                        (vẽ bounding box, telemetry text, v.v.)
                        Ví dụ: def my_overlay(frame): cv2.putText(...); return frame
        """
        import cv2

        def _store_frame(f: np.ndarray):
            self._latest = f

        self._handler.on_frame(_store_frame)
        await self._handler.start()

        try:
            while not self._stopped:
                frame = self._latest
                if frame is not None:
                    display = overlay_fn(frame.copy()) if overlay_fn else frame
                    cv2.imshow(self._title, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                await asyncio.sleep(0.01)  # yield cho event loop
        finally:
            await self._handler.stop()
            cv2.destroyAllWindows()
