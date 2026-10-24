"""
DRONEEXE Core - Base Handler
Kiểu dữ liệu dùng chung cho tất cả drone handlers.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import time


class DroneType(str, Enum):
    """Supported drone types."""
    ARDUPILOT = "ardupilot"
    PX4 = "px4"


class FlightMode(str, Enum):
    """Common flight modes across drone platforms."""
    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    GUIDED = "GUIDED"
    AUTO = "AUTO"
    RTL = "RTL"
    LAND = "LAND"
    TAKEOFF = "TAKEOFF"
    BRAKE = "BRAKE"


class ConnectionState(str, Enum):
    """Drone connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    ERROR = "error"


@dataclass
class TelemetryData:
    """Real-time telemetry from the drone."""
    timestamp: float = 0.0

    # Position
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0          # meters (relative)
    altitude_msl: float = 0.0      # meters (above sea level)

    # Local NED Position (EKF origin frame)
    local_x: float = 0.0          # meters (North)
    local_y: float = 0.0          # meters (East)
    local_z: float = 0.0          # meters (Down, âm = đang bay lên)
    local_valid: bool = False     # True khi nhận được toạ độ Local NED

    # Attitude
    roll: float = 0.0              # degrees
    pitch: float = 0.0             # degrees
    yaw: float = 0.0               # degrees (heading)

    # Velocity
    vx: float = 0.0                # m/s (North)
    vy: float = 0.0                # m/s (East)
    vz: float = 0.0                # m/s (Down)
    ground_speed: float = 0.0      # m/s
    air_speed: float = 0.0         # m/s

    # System
    battery_voltage: float = 0.0   # Volts
    battery_percent: int = 0       # 0-100
    battery_current: float = 0.0   # Amps

    # Status
    armed: bool = False
    flight_mode: str = ""
    gps_fix: int = 0               # 0=no fix, 2=2D, 3=3D
    gps_satellites: int = 0
    connection_state: ConnectionState = ConnectionState.DISCONNECTED

    # Hardware & Safety Telemetry
    rangefinder_distance: float = -1.0  # meters (-1.0 if not available / invalid)
    rangefinder_timestamp: float = 0.0  # Mốc thời gian nhận ToF / Rangefinder
    ekf_healthy: bool = False           # EKF3 convergence / filter health
    last_status_text: str = ""          # Latest STATUSTEXT message from autopilot
    local_timestamp: float = 0.0        # Mốc thời gian nhận LOCAL_POSITION_NED

    # Errors & Link Quality (Kiểm soát sai bit & mất gói)
    error_message: str = ""
    total_receive_errors: int = 0       # Số gói tin bị lỗi bit / CRC không hợp lệ
    total_packets_received: int = 0     # Tổng số gói tin MAVLink nhận thành công
    packet_loss_pct: float = 0.0        # Tỷ lệ phần trăm mất gói tin (%)

    # Compatibility aliases for local NED velocities
    @property
    def local_vx(self) -> float:
        return self.vx

    @property
    def local_vy(self) -> float:
        return self.vy

    @property
    def local_vz(self) -> float:
        return self.vz


@dataclass
class DroneCommand:
    """A command to send to the drone."""
    command: str
    params: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseDroneHandler(ABC):
    """
    Abstract base class for drone handlers.
    Defines the interface for connect, arm, takeoff, goto, etc.
    """

    def __init__(self, drone_type: DroneType):
        self.drone_type = drone_type
        self._telemetry = TelemetryData()
        self._is_connected = False
        self._telemetry_callbacks: list[Callable[[TelemetryData], None]] = []

    @property
    def telemetry(self) -> TelemetryData:
        return self._telemetry

    def get_telemetry(self) -> TelemetryData:
        """Alias method cho property telemetry (dùng bởi harness và swarm scripts)."""
        return self._telemetry

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def on_telemetry(self, callback: Callable[[TelemetryData], None]):
        """Register a callback for telemetry updates."""
        self._telemetry_callbacks.append(callback)

    def _notify_telemetry(self):
        """Notify all registered callbacks with current telemetry."""
        for cb in self._telemetry_callbacks:
            try:
                cb(self._telemetry)
            except Exception:
                pass

    # --- Connection ---
    @abstractmethod
    async def connect(self, connection_string: str) -> bool: ...

    @abstractmethod
    async def disconnect(self) -> bool: ...

    # --- Basic Flight Commands ---
    @abstractmethod
    async def arm(self, force: bool = False) -> bool: ...

    @abstractmethod
    async def disarm(self, force: bool = False) -> bool: ...

    @abstractmethod
    async def takeoff(self, altitude: float = 2.0) -> bool: ...

    @abstractmethod
    async def land(self) -> bool: ...

    @abstractmethod
    async def return_to_launch(self) -> bool: ...

    # --- Movement ---
    @abstractmethod
    async def goto(self, lat: float, lon: float, alt: float) -> bool: ...

    @abstractmethod
    async def move_relative(self, forward: float, right: float, down: float, yaw: float = 0) -> bool: ...

    @abstractmethod
    async def set_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0) -> bool: ...

    # --- Flight Mode ---
    @abstractmethod
    async def set_mode(self, mode: FlightMode) -> bool: ...

    @abstractmethod
    async def get_mode(self) -> FlightMode: ...

    # --- Telemetry ---
    @abstractmethod
    async def start_telemetry(self, rate_hz: int = 10) -> bool: ...

    @abstractmethod
    async def stop_telemetry(self) -> bool: ...

    # --- Emergency ---
    async def emergency_stop(self) -> bool:
        """Emergency stop — kill motors immediately."""
        return await self.land()
