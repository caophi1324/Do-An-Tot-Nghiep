"""
DRONEEXE Core - MAVLink Handler
Giao tiếp với ArduPilot/PX4 drone qua giao thức MAVLink.
Sử dụng thư viện pymavlink.
"""

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("mavlink_handler")

from .base_handler import (
    BaseDroneHandler,
    DroneType,
    FlightMode,
    ConnectionState,
    TelemetryData,
)


# MAVLink flight mode mappings for ArduPilot Copter
_ARDUPILOT_MODES = {
    FlightMode.STABILIZE: 0,
    FlightMode.ALT_HOLD: 2,
    FlightMode.LOITER: 5,
    FlightMode.GUIDED: 4,
    FlightMode.AUTO: 3,
    FlightMode.RTL: 6,
    FlightMode.LAND: 9,
    FlightMode.BRAKE: 17,
}

_ARDUPILOT_MODE_NAMES = {v: k for k, v in _ARDUPILOT_MODES.items()}


class MAVLinkHandler(BaseDroneHandler):
    """
    Drone handler for ArduPilot/PX4 via MAVLink protocol.
    Uses pymavlink for communication.
    """

    def __init__(self, drone_type: DroneType = DroneType.ARDUPILOT):
        super().__init__(drone_type)
        self._mavconn = None
        self._io_executor: Optional[ThreadPoolExecutor] = None
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._telemetry_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_paused: bool = False
        self._last_command_acks = {}
        self._parameters: Dict[str, float] = {}
        self._status_callbacks: list = []
        self._last_status_text: str = ""

    @property
    def last_status_text(self) -> str:
        return self._last_status_text

    def on_status_text(self, callback):
        """Register a callback for STATUSTEXT messages."""
        self._status_callbacks.append(callback)

    def pause_heartbeat(self):
        """Pause GCS heartbeat transmission (useful for testing GCS failsafe)."""
        self._heartbeat_paused = True
        logger.warning("GCS Heartbeat PAUSED (simulating link loss)")

    def resume_heartbeat(self):
        """Resume GCS heartbeat transmission."""
        self._heartbeat_paused = False
        logger.info("GCS Heartbeat RESUMED")

    # --- Connection ---

    async def connect(self, connection_string: str) -> bool:
        """
        Connect to ArduPilot/PX4 drone via MAVLink.
        Examples:
          - "udp:127.0.0.1:14550" (SITL default)
          - "tcp:127.0.0.1:5760"
          - "serial:COM3:57600"
        """
        try:
            from pymavlink import mavutil

            self._telemetry.connection_state = ConnectionState.CONNECTING
            self._notify_telemetry()

            # Parse connection string safely
            parts = connection_string.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid connection string '{connection_string}'. Format: proto:ip:port or serial:port:baud")
            proto = parts[0]

            if proto == "udp":
                mav_str = f"udpin:{parts[1]}:{parts[2]}" if len(parts) > 2 else parts[1]
            elif proto in ("udpout", "udpin"):
                mav_str = connection_string
            elif proto == "tcp":
                if len(parts) < 3:
                    raise ValueError(f"TCP connection string requires host and port, got: '{connection_string}'")
                mav_str = f"tcp:{parts[1]}:{parts[2]}"
            elif proto == "serial":
                baud = int(parts[2]) if len(parts) > 2 else 57600
                mav_str = parts[1]
                self._mavconn = mavutil.mavlink_connection(mav_str, baud=baud)
            else:
                raise ValueError(f"Unsupported protocol: {proto}")

            if proto != "serial":
                self._mavconn = mavutil.mavlink_connection(mav_str)

            # Triệt tiêu hoàn toàn độ trễ Nagle trên TCP socket (chống nghẽn gói tin gây khựng ở 20Hz)
            if hasattr(self._mavconn, 'port') and hasattr(self._mavconn.port, 'setsockopt'):
                try:
                    import socket
                    self._mavconn.port.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass

            # Wait for heartbeat
            logger.info(f"Waiting for heartbeat from {connection_string}...")
            loop = asyncio.get_event_loop()
            heartbeat = await loop.run_in_executor(
                None,
                lambda: self._mavconn.wait_heartbeat(timeout=10),
            )

            if heartbeat is None:
                logger.error("No heartbeat received — connection timeout")
                self._telemetry.connection_state = ConnectionState.ERROR
                self._telemetry.error_message = "No heartbeat received"
                self._notify_telemetry()
                return False

            self._is_connected = True
            self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mav-io-{self._mavconn.target_system}")
            self._telemetry.connection_state = ConnectionState.CONNECTED
            self._notify_telemetry()

            logger.info(
                f"Connected to {connection_string} "
                f"(system {self._mavconn.target_system}, "
                f"component {self._mavconn.target_component})"
            )

            # Request data streams
            self._request_data_streams()

            # Start GCS heartbeat sender (1Hz)
            self._start_heartbeat()

            # Start telemetry loop automatically at 20Hz matching control loop
            await self.start_telemetry(rate_hz=20)

            return True

        except Exception as e:
            logger.error(f"MAVLink connect error: {e}")
            self._telemetry.connection_state = ConnectionState.ERROR
            self._telemetry.error_message = str(e)
            self._notify_telemetry()
            return False

    async def disconnect(self) -> bool:
        await self._stop_heartbeat()
        await self.stop_telemetry()
        if self._mavconn:
            self._mavconn.close()
            self._mavconn = None
        if self._io_executor:
            self._io_executor.shutdown(wait=False)
            self._io_executor = None
        self._is_connected = False
        self._telemetry.connection_state = ConnectionState.DISCONNECTED
        self._notify_telemetry()
        logger.info("MAVLink disconnected")
        return True

    def _start_heartbeat(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_paused = False
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("GCS Heartbeat loop started (1Hz)")

    async def _stop_heartbeat(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("GCS Heartbeat loop stopped")

    async def _heartbeat_loop(self):
        """Background loop sending GCS heartbeat at 1Hz to maintain failsafe timers on FC."""
        from pymavlink import mavutil
        while self._is_connected and self._mavconn:
            try:
                if not self._heartbeat_paused:
                    async with self._send_lock:
                        self._mavconn.mav.heartbeat_send(
                            mavutil.mavlink.MAV_TYPE_GCS,
                            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                            0, 0, 0
                        )
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat send error: {e}")
                await asyncio.sleep(1.0)

    def _request_data_streams(self):
        """Request all data streams from the autopilot."""
        if not self._mavconn:
            return
        from pymavlink import mavutil

        # 1. Legacy stream request
        self._mavconn.mav.request_data_stream_send(
            self._mavconn.target_system,
            self._mavconn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,  # rate 20Hz
            1,   # start
        )

        # 2. Modern SET_MESSAGE_INTERVAL for ArduPilot 4.3+
        # (Local Pos 20Hz, Attitude 20Hz, Global Pos 5Hz, Rangefinder 20Hz, EKF Status 2Hz)
        streams = [
            (32, 50000),    # LOCAL_POSITION_NED (20Hz / 50ms)
            (30, 50000),    # ATTITUDE (20Hz / 50ms)
            (33, 200000),   # GLOBAL_POSITION_INT (5Hz)
            (173, 50000),   # RANGEFINDER (20Hz)
            (132, 50000),   # DISTANCE_SENSOR (20Hz)
            (193, 500000),  # EKF_STATUS_REPORT (2Hz)
        ]
        for msg_id, interval_us in streams:
            self._mavconn.mav.command_long_send(
                self._mavconn.target_system,
                self._mavconn.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                interval_us,
                0, 0, 0, 0,
                0  # param7 = 0: current link only (response target)
            )

        # 3. Đảm bảo WP_YAW_BEHAVIOR = 0 (Never change yaw automatically in guided mode)
        # Giúp drone bay chuyển vị đa hướng mượt mà, triệt tiêu 100% hiện tượng xoay lắc đầu/giật góc Yaw
        try:
            self._mavconn.mav.param_set_send(
                self._mavconn.target_system,
                self._mavconn.target_component,
                b"WP_YAW_BEHAVIOR",
                0.0,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
        except Exception as e:
            logger.warning(f"Could not set WP_YAW_BEHAVIOR: {e}")

    async def get_param(self, name: str, timeout: float = 3.0) -> Optional[float]:
        """
        Request and wait for a parameter value from the autopilot.
        Returns float value if received, None if timed out or parameter does not exist.
        """
        if not self._mavconn:
            return None
        sys_id = self._mavconn.target_system or 1
        comp_id = self._mavconn.target_component or 1
        param_id_bytes = name.encode('utf-8')[:16]

        self._parameters.pop(name, None)

        async with self._send_lock:
            self._mavconn.mav.param_request_read_send(
                sys_id, comp_id,
                param_id_bytes,
                -1
            )

        start_t = time.time()
        while time.time() - start_t < timeout:
            if name in self._parameters:
                return self._parameters[name]
            await asyncio.sleep(0.05)
        return None

    # --- Basic Flight Commands ---

    async def arm(self, force: bool = False) -> bool:
        if not self._mavconn:
            return False
        if self._telemetry.armed:
            return True
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            # Đặt GUIDED mode trước khi Arm
            await self.set_mode(FlightMode.GUIDED)

            self._last_command_acks.pop(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, None)
            # param1=1 (Arm), param2: 0 = strict safety check, 21196 = force arm
            magic_param2 = 21196 if force else 0
            if force:
                logger.warning(f"[Sys {sys_id}] Gửi lệnh FORCE ARM (param2=21196) theo yêu cầu!")
            else:
                logger.info(f"[Sys {sys_id}] Gửi lệnh ARM nghiêm ngặt (param2=0, tuân thủ ARMING_CHECK)")

            self._mavconn.mav.command_long_send(
                sys_id,
                comp_id,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, magic_param2, 0, 0, 0, 0, 0,
            )

            ack = await self._wait_command_ack(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                timeout=4.0,
            )
            # Chờ xác nhận trạng thái armed thật sự từ Autopilot qua HEARTBEAT
            for _ in range(15):
                if self._telemetry.armed:
                    logger.info(f"[Sys {sys_id}] Drone armed successfully (xác nhận từ HEARTBEAT)")
                    return True
                await asyncio.sleep(0.2)

            if not self._telemetry.armed:
                logger.warning(f"[Sys {sys_id}] Lệnh ARM không kích hoạt được động cơ (ACK={ack}, Status: {self._last_status_text})")
                return False
            return True

        except Exception as e:
            logger.error(f"Arm error: {e}")
            return False

    async def disarm(self, force: bool = False) -> bool:
        if not self._mavconn:
            return False
        if not self._telemetry.armed:
            return True
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            # An toàn phần cứng: Xác thực tiếp đất trước khi cho phép ngắt động cơ nếu force=False
            if not force:
                now = time.time()
                rng_fresh = (now - self._telemetry.rangefinder_timestamp) < 1.5
                loc_fresh = (now - self._telemetry.local_timestamp) < 1.5
                ground_confirmed = False
                if rng_fresh and 0.0 <= self._telemetry.rangefinder_distance < 0.25:
                    ground_confirmed = True
                elif loc_fresh and abs(self._telemetry.altitude) < 0.25:
                    ground_confirmed = True

                if not ground_confirmed:
                    logger.warning(
                        f"[Sys {sys_id}] TỪ CHỐI DISARM: Drone chưa xác nhận tiếp đất an toàn "
                        f"(Alt={self._telemetry.altitude:.2f}m, ToF={self._telemetry.rangefinder_distance:.2f}m, "
                        f"RngFresh={rng_fresh}, LocFresh={loc_fresh}). "
                        f"Hãy gọi land() hoặc truyền force=True nếu khẩn cấp."
                    )
                    return False

            self._last_command_acks.pop(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, None)
            magic_param2 = 21196 if force else 0
            async with self._send_lock:
                self._mavconn.mav.command_long_send(
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, magic_param2, 0, 0, 0, 0, 0,
                )

            ack = await self._wait_command_ack(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
            )
            # Chờ xác nhận từ Heartbeat
            for _ in range(15):
                if not self._telemetry.armed:
                    self._telemetry.connection_state = ConnectionState.CONNECTED
                    self._notify_telemetry()
                    logger.info(f"[Sys {sys_id}] Drone disarmed thành công (xác nhận từ HEARTBEAT)")
                    return True
                await asyncio.sleep(0.1)

            return ack and not self._telemetry.armed

        except Exception as e:
            logger.error(f"Disarm error: {e}")
            return False

    async def takeoff(self, altitude: float = 2.0) -> bool:
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            await self.set_mode(FlightMode.GUIDED)

            # Retry arming up to 8 times (cho phep EKF on dinh ~12s)
            for attempt in range(8):
                if not self._telemetry.armed:
                    await self.arm()
                    if self._telemetry.armed:
                        break
                    logger.warning(f"[Sys {sys_id}] Arm attempt {attempt+1} pending (cho EKF khoi tao). Thu lai sau 1.5s...")
                    await asyncio.sleep(1.5)
            
            if not self._telemetry.armed:
                logger.error(f"[Sys {sys_id}] Khong the cat canh vi drone chua ARM duoc!")
                return False

            await asyncio.sleep(1.0) # Chờ động cơ quay đều

            # Retry takeoff command up to 4 times
            ack = False
            threshold_alt = max(0.3, altitude * 0.5)
            for t_attempt in range(4):
                self._last_command_acks.pop(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, None)
                self._mavconn.mav.command_long_send(
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0, 0, 0, 0, 0, 0, 0, altitude,
                )
                ack = await self._wait_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=3.0)
                if ack or self._telemetry.altitude >= threshold_alt:
                    ack = True
                    break
                logger.warning(f"[Sys {sys_id}] Takeoff attempt {t_attempt+1} pending. Retrying in 1.5s...")
                await asyncio.sleep(1.5)

            if ack or self._telemetry.altitude >= threshold_alt:
                self._telemetry.connection_state = ConnectionState.FLYING
                self._notify_telemetry()
                logger.info(f"[Sys {sys_id}] Taking off to {altitude}m")
                return True
            return False

        except Exception as e:
            logger.error(f"Takeoff error: {e}")
            return False

    async def land(self) -> bool:
        return await self.set_mode(FlightMode.LAND)


    async def return_to_launch(self) -> bool:
        return await self.set_mode(FlightMode.RTL)

    # --- Movement ---

    async def goto(self, lat: float, lon: float, alt: float) -> bool:
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil

            await self.set_mode(FlightMode.GUIDED)

            self._mavconn.mav.set_position_target_global_int_send(
                0,
                self._mavconn.target_system,
                self._mavconn.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
                0, 0, 0,
                0, 0, 0,
                0, 0,
            )
            logger.info(f"Going to ({lat:.6f}, {lon:.6f}) at {alt}m")
            return True

        except Exception as e:
            logger.error(f"Goto error: {e}")
            return False

    async def set_position_local(
        self, x: float, y: float, z_alt: float,
        yaw: float = None,
        vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
        use_vel: bool = False
    ) -> bool:
        """
        Di chuyển drone đến toạ độ TUYỆT ĐỐI (X: North, Y: East, Z: Altitude) trong hệ toạ độ Local NED.
        z_alt: Độ cao dương tính bằng mét (ví dụ 5.0m -> Z_NED = -5.0m).
        yaw: Góc hướng đầu drone tính bằng độ (0=North, 90=East).
             Nếu yaw is None: Bỏ qua điều khiển yaw giúp drone giữ nguyên hướng tự nhiên, triệt tiêu giật đầu.
        use_vel: Nếu True, truyền véc-tơ vận tốc feedforward (vx, vy, vz) qua mask 0b0000111111000000.
                 ArduPilot sẽ kết hợp vị trí + vận tốc, triệt tiêu 100% hiện tượng phanh khựng / giật cục.
        """
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            if use_vel:
                if yaw is None:
                    type_mask = 0b0000111111000000  # Dùng Pos + Vel, bỏ qua Accel, bỏ qua Yaw
                    yaw_rad = 0.0
                else:
                    type_mask = 0b0000101111000000  # Dùng Pos + Vel + Yaw, bỏ qua Accel
                    yaw_rad = math.radians(yaw)
            else:
                if yaw is None:
                    type_mask = 0b0000111111111000  # Chỉ dùng Pos, bỏ qua Vel + Accel + Yaw
                    yaw_rad = 0.0
                else:
                    type_mask = 0b0000101111111000  # Dùng Pos + Yaw, bỏ qua Vel + Accel
                    yaw_rad = math.radians(yaw)

            async with self._send_lock:
                self._mavconn.mav.set_position_target_local_ned_send(
                    0,
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    type_mask,
                    x, y, -z_alt,
                    vx, vy, vz,
                    0, 0, 0,
                    yaw_rad, 0,
                )
            return True
        except Exception as e:
            logger.error(f"Set position local error: {e}")
            return False

    async def move_relative(self, forward: float, right: float, down: float, yaw: float = None) -> bool:
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            if yaw is None:
                type_mask = 0b0000111111111000
                yaw_rad = 0.0
            else:
                type_mask = 0b0000101111111000
                yaw_rad = math.radians(yaw)

            # BODY_OFFSET_NED: ArduPilot tự xoay theo heading drone
            async with self._send_lock:
                self._mavconn.mav.set_position_target_local_ned_send(
                    0,
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    type_mask,
                    forward, right, down,
                    0, 0, 0,
                    0, 0, 0,
                    yaw_rad, 0,
                )
            return True

        except Exception as e:
            logger.error(f"Move relative error: {e}")
            return False

    async def set_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0) -> bool:
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            # MAV_FRAME_BODY_NED: Khung chuẩn khuyến nghị cho velocity
            async with self._send_lock:
                self._mavconn.mav.set_position_target_local_ned_send(
                    0,
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,
                    0b0000011111000111,  # bit 11 cleared → enable yaw_rate
                    0, 0, 0,
                    vx, vy, vz,
                    0, 0, 0,
                    0, math.radians(yaw_rate),
                )
            return True

        except Exception as e:
            logger.error(f"Set velocity error: {e}")
            return False


    # --- Flight Mode ---

    async def set_mode(self, mode: FlightMode) -> bool:
        if not self._mavconn:
            return False
        # KHÔNG early-return đối với LAND và RTL để bảo đảm tính cưỡng bức an toàn
        if self._telemetry.flight_mode == mode.value and mode not in (FlightMode.LAND, FlightMode.RTL):
            return True
        try:
            from pymavlink import mavutil
            sys_id = self._mavconn.target_system or 1
            comp_id = self._mavconn.target_component or 1

            mode_id = _ARDUPILOT_MODES.get(mode)
            if mode_id is None:
                logger.warning(f"Unsupported mode: {mode}")
                return False

            # Gửi set_mode_send và MAV_CMD_DO_SET_MODE
            async with self._send_lock:
                self._mavconn.mav.set_mode_send(
                    sys_id,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id,
                )
                self._mavconn.mav.command_long_send(
                    sys_id,
                    comp_id,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id,
                    0, 0, 0, 0, 0,
                )

            # Chờ xác nhận từ Heartbeat (tối đa 3.5s)
            for _ in range(35):
                if self._telemetry.flight_mode == mode.value:
                    logger.info(f"[Sys {sys_id}] Mode set to {mode.value}")
                    return True
                await asyncio.sleep(0.1)

            logger.warning(f"[Sys {sys_id}] Mode change to {mode.value} not confirmed (Current: {self._telemetry.flight_mode})")
            return False

        except Exception as e:
            logger.error(f"Set mode error: {e}")
            return False


    async def get_mode(self) -> FlightMode:
        """Đọc mode từ telemetry nội bộ được cập nhật liên tục từ HEARTBEAT (không tranh chấp recv_match)."""
        if self._telemetry.flight_mode:
            for m in FlightMode:
                if m.value == self._telemetry.flight_mode:
                    return m
        return FlightMode.STABILIZE

    # --- Telemetry ---

    async def start_telemetry(self, rate_hz: int = 10) -> bool:
        if self._telemetry_task and not self._telemetry_task.done():
            return True
        self._telemetry_task = asyncio.create_task(self._telemetry_loop(rate_hz))
        logger.info(f"Telemetry started at {rate_hz}Hz")
        return True

    async def stop_telemetry(self) -> bool:
        if self._telemetry_task:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
            self._telemetry_task = None
            logger.info("Telemetry stopped")
        return True

    async def _telemetry_loop(self, rate_hz: int):
        """Background loop reading MAVLink messages and updating telemetry (Single Reader)."""
        interval = 1.0 / rate_hz
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Đọc toàn bộ message đang chờ trong buffer (giới hạn tối đa 300 msg/pass để tránh nghẽn thread)
                def _drain_messages():
                    msgs = []
                    if not self._mavconn:
                        return msgs
                    while True:
                        m = self._mavconn.recv_match(blocking=False)
                        if m is None:
                            break
                        msgs.append(m)
                        if len(msgs) >= 300:
                            break
                    return msgs

                messages = await loop.run_in_executor(self._io_executor, _drain_messages)

                for msg in messages:
                    msg_type = msg.get_type()
                    self._telemetry.timestamp = time.time()

                    if msg_type == "LOCAL_POSITION_NED":
                        self._telemetry.local_x = msg.x
                        self._telemetry.local_y = msg.y
                        self._telemetry.local_z = msg.z
                        self._telemetry.local_valid = True
                        self._telemetry.local_timestamp = time.time()

                    elif msg_type == "GLOBAL_POSITION_INT":
                        self._telemetry.latitude = msg.lat / 1e7
                        self._telemetry.longitude = msg.lon / 1e7
                        self._telemetry.altitude = msg.relative_alt / 1000.0
                        self._telemetry.altitude_msl = msg.alt / 1000.0  # Nguồn MSL duy nhất
                        self._telemetry.vx = msg.vx / 100.0
                        self._telemetry.vy = msg.vy / 100.0
                        self._telemetry.vz = msg.vz / 100.0

                    elif msg_type == "ATTITUDE":
                        self._telemetry.roll = math.degrees(msg.roll)
                        self._telemetry.pitch = math.degrees(msg.pitch)
                        self._telemetry.yaw = math.degrees(msg.yaw)

                    elif msg_type == "VFR_HUD":
                        self._telemetry.ground_speed = msg.groundspeed
                        self._telemetry.air_speed = msg.airspeed
                        # Không ghi đè altitude_msl từ VFR_HUD để tránh giật số liệu

                    elif msg_type == "SYS_STATUS":
                        self._telemetry.battery_voltage = msg.voltage_battery / 1000.0
                        self._telemetry.battery_current = msg.current_battery / 100.0
                        self._telemetry.battery_percent = msg.battery_remaining

                    elif msg_type == "GPS_RAW_INT":
                        self._telemetry.gps_fix = msg.fix_type
                        self._telemetry.gps_satellites = msg.satellites_visible

                    elif msg_type == "DISTANCE_SENSOR":
                        # ToF khoảng cách từ MTF-02 (đơn vị cm -> mét)
                        if getattr(msg, "orientation", 25) == 25:
                            self._telemetry.rangefinder_distance = msg.current_distance / 100.0
                            self._telemetry.rangefinder_timestamp = time.time()

                    elif msg_type == "RANGEFINDER":
                        self._telemetry.rangefinder_distance = float(msg.distance)
                        self._telemetry.rangefinder_timestamp = time.time()

                    elif msg_type == "EKF_STATUS_REPORT":
                        # Cờ EKF3 hội tụ: 4 bit bắt buộc (attitude=1, vel_horiz=2, vel_vert=4, pos_horiz_rel=8 -> tổng 0x0F)
                        # và cờ cảnh báo EKF_UNINITIALIZED (bit 10 = 1024) không được bật
                        flags = getattr(msg, "flags", 0)
                        vel_var = getattr(msg, "velocity_variance", 0.0)
                        pos_var = getattr(msg, "pos_horiz_variance", 0.0)
                        has_all_flags = (flags & 0x0F) == 0x0F
                        not_uninit = not bool(flags & 1024)
                        self._telemetry.ekf_healthy = bool(has_all_flags and not_uninit and vel_var < 1.0 and pos_var < 1.0)
                        self._telemetry.ekf_timestamp = time.time()

                    elif msg_type == "STATUSTEXT":
                        text = getattr(msg, "text", "")
                        if isinstance(text, bytes):
                            text = text.decode("utf-8", errors="ignore")
                        text = text.strip('\x00').strip()
                        if text:
                            self._last_status_text = text
                            self._telemetry.last_status_text = text
                            for cb in self._status_callbacks:
                                try:
                                    cb(text)
                                except Exception as cb_err:
                                    logger.debug(f"Status callback error: {cb_err}")

                    elif msg_type == "HEARTBEAT":
                        from pymavlink import mavutil
                        self._telemetry.armed = (
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        ) != 0
                        mode = _ARDUPILOT_MODE_NAMES.get(msg.custom_mode)
                        if mode:
                            self._telemetry.flight_mode = mode.value

                        if self._telemetry.armed:
                            self._telemetry.connection_state = ConnectionState.ARMED
                        else:
                            self._telemetry.connection_state = ConnectionState.CONNECTED

                    elif msg_type == "COMMAND_ACK":
                        self._last_command_acks[msg.command] = msg.result

                    elif msg_type == "PARAM_VALUE":
                        p_id = getattr(msg, "param_id", "")
                        if isinstance(p_id, bytes):
                            p_id = p_id.decode("utf-8", errors="ignore")
                        p_id = p_id.strip('\x00').strip()
                        self._parameters[p_id] = float(getattr(msg, "param_value", 0.0))

                if self._mavconn:
                    if hasattr(self._mavconn, 'mav'):
                        self._telemetry.total_receive_errors = getattr(self._mavconn.mav, 'total_receive_errors', 0)
                        self._telemetry.total_packets_received = getattr(self._mavconn.mav, 'total_packets_received', 0)
                    if hasattr(self._mavconn, 'packet_loss'):
                        try:
                            self._telemetry.packet_loss_pct = float(self._mavconn.packet_loss())
                        except Exception:
                            pass

                if messages:
                    self._notify_telemetry()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                await asyncio.sleep(0.5)

    # --- Emergency ---

    async def emergency_stop(self) -> bool:
        """
        Emergency stop / Emergency Land:
        Chuyển sang chế độ LAND cưỡng bức. Tuyệt đối KHÔNG disarm giữa không trung khi
        telemetry chưa xác thực hoặc mất link. Giám sát liên tục cho tới khi drone disarm thật.
        """
        if not self._mavconn:
            return False
        try:
            from pymavlink import mavutil

            logger.warning("EMERGENCY LAND ACTIVATED — Phát lệnh cưỡng bức hạ cánh (LAND)!")

            # 1. Luôn gửi lệnh LAND cưỡng bức
            await self.set_mode(FlightMode.LAND)

            # 2. Vòng lặp giám sát hạ cánh an toàn (Landing Watchdog tối đa 30 giây)
            start_time = time.time()
            while time.time() - start_time < 30.0:
                if not self._telemetry.armed:
                    logger.info("Đã xác nhận động cơ đã ngắt (DISARMED) an toàn.")
                    return True

                # Kiểm tra độ tươi của telemetry từng nguồn (dưới 1.5s)
                now = time.time()
                rng_fresh = (now - self._telemetry.rangefinder_timestamp) < 1.5
                loc_fresh = (now - self._telemetry.local_timestamp) < 1.5

                # Kiểm tra nguồn độ cao xác thực
                ground_confirmed = False
                if rng_fresh and 0.0 <= self._telemetry.rangefinder_distance < 0.25:
                    ground_confirmed = True
                elif loc_fresh and abs(self._telemetry.altitude) < 0.25:
                    ground_confirmed = True

                if ground_confirmed:
                    # Đã tiếp đất chắc chắn: ngắt motor an toàn
                    self._last_command_acks.pop(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, None)
                    async with self._send_lock:
                        self._mavconn.mav.command_long_send(
                            self._mavconn.target_system,
                            self._mavconn.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0, 0, 21196, 0, 0, 0, 0, 0,
                        )
                    logger.warning("Tiếp đất xác nhận — Đã ngắt động cơ khẩn cấp an toàn.")
                    await asyncio.sleep(0.5)
                    if not self._telemetry.armed:
                        return True

                await asyncio.sleep(0.5)

            logger.warning(f"Hết thời gian giám sát hạ cánh khẩn cấp (30s). Drone armed={self._telemetry.armed}")
            return not self._telemetry.armed
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
            return False

    async def emergency_land(self) -> bool:
        """Bí danh an toàn cho emergency_stop() để phản ánh đúng bản chất hoạt động."""
        return await self.emergency_stop()

    # --- Helpers ---

    async def _wait_command_ack(self, command_id: int, timeout: float = 5.0) -> bool:
        """Wait for a COMMAND_ACK message, handling MAV_RESULT_IN_PROGRESS properly."""
        try:
            from pymavlink import mavutil
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                if command_id in self._last_command_acks:
                    result = self._last_command_acks[command_id]
                    if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        self._last_command_acks.pop(command_id, None)
                        return True
                    elif result == getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5):
                        logger.info(f"Command {command_id} in progress... continuing to wait.")
                        await asyncio.sleep(0.2)
                        continue
                    else:
                        self._last_command_acks.pop(command_id, None)
                        logger.warning(f"Command {command_id} rejected: result={result}")
                        return False
                await asyncio.sleep(0.05)
                
            logger.warning(f"Timeout waiting for ACK for command {command_id}")
            return False
            
        except Exception as e:
            logger.error(f"Wait ACK error: {e}")
            return False
