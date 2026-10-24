"""Core processing layer for Face Recognition & Object Tracking System."""
import os
try:
    import uniface
    local_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    if os.path.exists(local_models):
        uniface.set_cache_dir(local_models)
except Exception:
    pass

from .config import AppConfig, TrackingMode, SystemMode, OverlayConfig
from .smoothing import OneEuro, xyxy_to_cxcywh, cxcywh_to_xyxy, BoxSmoother
from .quality import filter_5_landmarks, score_face_quality, FaceQualityEstimator
from .detectors import PersonDetector, FaceDetector, PersonDetection, FaceDetectionResult
from .tracker import TrackManager, TrackState, TrackResult
from .recognizer import FaceRecognizer, align_face_112
from .sources import VideoSource, WebcamSource, StreamSource, FileSource, open_source
from .pipeline import Pipeline, render_overlay
from .runner import PipelineRunner

__all__ = [
    "AppConfig",
    "TrackingMode",
    "SystemMode",
    "OverlayConfig",
    "OneEuro",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "BoxSmoother",
    "filter_5_landmarks",
    "score_face_quality",
    "FaceQualityEstimator",
    "PersonDetector",
    "FaceDetector",
    "PersonDetection",
    "FaceDetectionResult",
    "TrackManager",
    "TrackState",
    "TrackResult",
    "FaceRecognizer",
    "align_face_112",
    "VideoSource",
    "WebcamSource",
    "StreamSource",
    "FileSource",
    "open_source",
    "Pipeline",
    "render_overlay",
    "PipelineRunner",
]
