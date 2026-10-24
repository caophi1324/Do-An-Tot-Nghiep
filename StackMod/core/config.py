"""Configuration data structures and presets for the system."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class TrackingMode(str, Enum):
    PERSON = "person"
    FACE = "face"


class SystemMode(str, Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"
    ENROLL = "enroll"
    PLAYBACK = "playback"
    MONITOR = "monitor"


class ExecutionProvider(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    DIRECTML = "directml"
    CPU = "cpu"


import logging

logger = logging.getLogger(__name__)

# Disable Ultralytics auto-installing cpu-only onnxruntime
os.environ["YOLO_AUTOINSTALL"] = "0"


_DETECTED_GPU_NAME: Optional[str] = None


def get_gpu_device_name(device_id: Optional[int] = None) -> str:
    """Returns detected GPU model name across NVIDIA, AMD, and Intel GPUs.
    
    If device_id is specified, returns that adapter's name. Otherwise, prioritizes dedicated GPU.
    """
    global _DETECTED_GPU_NAME
    if device_id is None and _DETECTED_GPU_NAME is not None:
        return _DETECTED_GPU_NAME

    # 1. Try PyTorch CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            dev_idx = device_id if device_id is not None else 0
            name = str(torch.cuda.get_device_name(dev_idx)).strip()
            if name:
                if device_id is None:
                    _DETECTED_GPU_NAME = name
                return name
    except Exception:
        pass

    # 2. Try Windows Registry
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
            )
            # If specific device_id requested, query directly
            if device_id is not None:
                try:
                    sub = winreg.OpenKey(key, f"{int(device_id):04d}")
                    desc, _ = winreg.QueryValueEx(sub, "DriverDesc")
                    desc_str = str(desc).strip()
                    if desc_str and "Basic" not in desc_str:
                        return desc_str
                except Exception:
                    pass

            # Otherwise, scan all adapters and prioritize dedicated GPU (NVIDIA / AMD / RTX)
            found_gpus: list[tuple[int, str]] = []
            for i in range(16):
                try:
                    sub = winreg.OpenKey(key, f"{i:04d}")
                    desc, _ = winreg.QueryValueEx(sub, "DriverDesc")
                    desc_str = str(desc).strip()
                    if desc_str and "Basic" not in desc_str and "Remote" not in desc_str:
                        found_gpus.append((i, desc_str))
                except Exception:
                    pass

            # Prioritize dedicated GPU
            for idx, gname in found_gpus:
                up = gname.upper()
                if "NVIDIA" in up or "RTX" in up or "GEFORCE" in up or "RADEON" in up:
                    _DETECTED_GPU_NAME = gname
                    return gname

            if found_gpus:
                _DETECTED_GPU_NAME = found_gpus[0][1]
                return _DETECTED_GPU_NAME
        except Exception:
            pass

    _DETECTED_GPU_NAME = "GPU"
    return _DETECTED_GPU_NAME


def get_optimal_device_id() -> int:
    """Returns detected optimal GPU device index (preferring dedicated NVIDIA/AMD GPUs over integrated GPUs)."""
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
            )
            for i in range(16):
                try:
                    sub = winreg.OpenKey(key, f"{i:04d}")
                    desc, _ = winreg.QueryValueEx(sub, "DriverDesc")
                    desc_str = str(desc).strip().upper()
                    if ("NVIDIA" in desc_str or "RADEON" in desc_str or "RTX" in desc_str or "GEFORCE" in desc_str) and "BASIC" not in desc_str:
                        return i
                except Exception:
                    pass
        except Exception:
            pass
    return 0


def get_dml_session_options(device_id: int = 0) -> tuple[Any, list[dict[str, Any]]]:
    """Configures ONNX Runtime SessionOptions and provider_options following Microsoft DirectML specifications.

    Official DirectML EP Best Practices:
    - enable_mem_pattern = False (DirectML does not support memory pattern optimization).
    - execution_mode = ORT_SEQUENTIAL (DirectML does not support parallel intra-op execution).
    - graph_optimization_level = ORT_ENABLE_ALL (Enables constant folding and op fusing).
    - provider_options with device_id (Maps to specific DirectX 12 adapter, e.g. RX 7600, RTX A2000).
    """
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options = [{"device_id": str(device_id)}, {}]
        return opts, provider_options
    except Exception as e:
        logger.warning(f"[Hardware] Could not create DML SessionOptions: {e}")
        return None, []


def get_cuda_session_options(device_id: int = 0) -> tuple[Any, list[dict[str, Any]]]:
    """Configures ONNX Runtime SessionOptions and provider_options for NVIDIA CUDA."""
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options = [{"device_id": str(device_id), "arena_extend_strategy": "kNextPowerOfTwo"}, {}]
        return opts, provider_options
    except Exception as e:
        logger.warning(f"[Hardware] Could not create CUDA SessionOptions: {e}")
        return None, []


def log_active_providers(model_name: str, session: Any) -> list[str]:
    """Logs the active execution providers actually used by an ONNX session."""
    if session is not None and hasattr(session, "get_providers"):
        active = session.get_providers()
        logger.info(f"[Hardware] {model_name} active ONNX providers: {active}")
        return active
    return []


logger.info(
    "[Hardware] Uniface models (RetinaFace, EdgeFace, eDifFIQA) run using standard ONNX Runtime providers. "
    "Note: Switching hardware execution providers requires restarting the application or re-initializing models."
)


def get_optimal_providers(preference: str | ExecutionProvider = "auto") -> list[str]:
    """Auto-detects or configures ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        logger.info(f"[Hardware] Available ONNX providers: {available}")
    except Exception:
        logger.warning("[Hardware] ONNX Runtime not available, falling back to CPU")
        return ["CPUExecutionProvider"]

    if hasattr(preference, "value"):
        pref_str = str(preference.value).lower()
    else:
        pref_str = str(preference).lower()
    if "." in pref_str:
        pref_str = pref_str.split(".")[-1]

    if pref_str == "cuda":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"[Hardware] Selected providers: {providers}")
            return providers
        logger.warning(
            "[Hardware] CUDA requested but CUDAExecutionProvider not found in ONNX Runtime. "
            "Falling back to auto-detection (DirectML / CPU)."
        )
    elif pref_str == "directml":
        if "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"[Hardware] Selected providers: {providers}")
            return providers
        logger.warning(
            "[Hardware] DirectML requested but DmlExecutionProvider not found in ONNX Runtime. "
            "Falling back to auto-detection."
        )
    elif pref_str == "cpu":
        logger.info("[Hardware] Selected providers: ['CPUExecutionProvider']")
        return ["CPUExecutionProvider"]

    # AUTO Detection: Prioritize CUDA -> DirectML (AMD/Intel/Windows GPU) -> CoreML -> CPU
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    logger.info(f"[Hardware] Selected providers: {providers}")
    return providers


@dataclass
class HardwareConfig:
    provider: ExecutionProvider = ExecutionProvider.AUTO
    device_id: int = field(default_factory=get_optimal_device_id)


@dataclass
class PersonDetectionConfig:
    enabled: bool = True
    model_path: str = "yolo11n.pt"
    conf: float = 0.45
    iou: float = 0.45
    half: bool = False
    person_class_id: int = 0
    skip_frames: int = 0  # 0 means process every frame
    device_id: int = field(default_factory=get_optimal_device_id)


@dataclass
class FaceDetectionConfig:
    enabled: bool = True
    model_name: str = "retinaface_mnet_v2"
    conf: float = 0.6
    nms: float = 0.4
    crop_upper_ratio: float = 0.6  # crop upper 60% of person bbox for face detection
    min_face_size: int = 50  # px


@dataclass
class TrackingConfig:
    mode: TrackingMode = TrackingMode.PERSON
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    low_thresh: float = 0.1


@dataclass
class LandmarkFilterConfig:
    enabled: bool = True
    min_face_size: float = 36.0  # px
    max_yaw_ratio: float = 0.45  # |nose_offset / eye_distance| <= 0.45
    max_roll_deg: float = 35.0   # roll angle in degrees
    min_pitch_ratio: float = 0.5 # (eye-to-nose) / (nose-to-mouth)
    max_pitch_ratio: float = 1.8
    min_conf: float = 0.50


@dataclass
class FaceQualityConfig:
    enabled: bool = True
    min_quality: float = 0.30
    upgrade_delta: float = 0.05  # only replace best face if new quality is higher by this delta
    method: str = "ediffiqa"


@dataclass
class RecognitionConfig:
    enabled: bool = True
    model_name: str = "edgeface_xxs"
    match_thr: float = 0.50      # Cosine similarity threshold for FAISS search (optimized for EdgeFace)
    vote_window: int = 5         # Number of recent frames considered (5 = ~0.17s @ 30fps, snappier re-ID)
    vote_min: int = 2            # Minimum identical votes to confirm identity
    hysteresis_thr: float = 0.42 # Retain identity threshold if already confirmed
    recheck_interval_s: float = 0.5  # Time between re-verification checks (0.5s = fast re-ID on re-entry)


@dataclass
class SmoothingConfig:
    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.01
    d_cutoff: float = 1.0
    hold_lost_frames: int = 5    # Draw predicted box for N frames after loss


@dataclass
class OverlayConfig:
    show_person_bbox: bool = True
    show_face_bbox: bool = True
    show_landmarks: bool = False
    show_id: bool = True
    show_name: bool = True
    show_quality: bool = True
    show_fps: bool = True
    show_latency: bool = True
    show_status_badge: bool = True


@dataclass
class StorageConfig:
    db_url: str = field(default_factory=lambda: os.environ.get("DB_URL") or os.environ.get("DATABASE_URL") or "sqlite:///data/face_app.db")
    faiss_path: str = field(default_factory=lambda: os.environ.get("FAISS_PATH", "data/faiss_index"))
    snapshots_dir: str = field(default_factory=lambda: os.environ.get("SNAPSHOTS_DIR", "data/snapshots"))
    models_dir: str = "models"
    recordings_dir: str = field(default_factory=lambda: os.environ.get("RECORDINGS_DIR", "data/recordings"))


@dataclass
class AppConfig:
    mode: SystemMode = SystemMode.SIMPLE
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    person_det: PersonDetectionConfig = field(default_factory=PersonDetectionConfig)
    face_det: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    landmark_filter: LandmarkFilterConfig = field(default_factory=LandmarkFilterConfig)
    quality: FaceQualityConfig = field(default_factory=FaceQualityConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        config = cls()
        if "mode" in data:
            config.mode = SystemMode(data["mode"])
        if "hardware" in data:
            h_data = data["hardware"].copy()
            if "provider" in h_data:
                h_data["provider"] = ExecutionProvider(h_data["provider"])
            config.hardware = HardwareConfig(**h_data)
        if "person_det" in data:
            config.person_det = PersonDetectionConfig(**data["person_det"])
        if "face_det" in data:
            config.face_det = FaceDetectionConfig(**data["face_det"])
        if "tracking" in data:
            t_data = data["tracking"].copy()
            if "mode" in t_data:
                t_data["mode"] = TrackingMode(t_data["mode"])
            config.tracking = TrackingConfig(**t_data)
        if "landmark_filter" in data:
            config.landmark_filter = LandmarkFilterConfig(**data["landmark_filter"])
        if "quality" in data:
            config.quality = FaceQualityConfig(**data["quality"])
        if "recognition" in data:
            config.recognition = RecognitionConfig(**data["recognition"])
        if "smoothing" in data:
            config.smoothing = SmoothingConfig(**data["smoothing"])
        if "overlay" in data:
            config.overlay = OverlayConfig(**data["overlay"])
        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])
        return config

    def save_json(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: str) -> AppConfig:
        if not os.path.exists(file_path):
            return cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def apply_preset(self, mode: SystemMode) -> None:
        self.mode = mode
        if mode == SystemMode.SIMPLE:
            self.overlay.show_landmarks = False
            self.overlay.show_quality = False
            self.overlay.show_face_bbox = True
            self.overlay.show_person_bbox = True
            self.tracking.mode = TrackingMode.PERSON
        elif mode == SystemMode.ADVANCED:
            self.overlay.show_landmarks = True
            self.overlay.show_quality = True
            self.overlay.show_face_bbox = True
            self.overlay.show_person_bbox = True
        elif mode == SystemMode.ENROLL:
            self.overlay.show_landmarks = True
            self.overlay.show_quality = True
            self.overlay.show_face_bbox = True
            self.overlay.show_person_bbox = False
            self.tracking.mode = TrackingMode.FACE
        elif mode == SystemMode.MONITOR:
            self.overlay.show_landmarks = False
            self.overlay.show_quality = True
            self.overlay.show_face_bbox = True
            self.overlay.show_person_bbox = True
            self.tracking.mode = TrackingMode.PERSON
        elif mode == SystemMode.PLAYBACK:
            self.overlay.show_landmarks = True
            self.overlay.show_quality = True
