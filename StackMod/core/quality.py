"""Face landmark filtering and face quality evaluation module."""
from __future__ import annotations

import math
import time
import logging
from typing import Optional, Tuple, Union
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    from uniface.quality import EDifFIQA, QualityResult
except ImportError:
    EDifFIQA = None
    QualityResult = None

from .config import LandmarkFilterConfig, FaceQualityConfig


def calculate_landmark_geometry(landmarks: np.ndarray) -> dict:
    """Computes geometric pose metrics from 5 facial landmarks.
    
    Landmark indexing:
        0: Left Eye
        1: Right Eye
        2: Nose
        3: Left Mouth Corner
        4: Right Mouth Corner
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (5, 2):
        return {
            "valid": False,
            "yaw_ratio": 1.0,
            "roll_deg": 90.0,
            "pitch_ratio": 0.0,
            "eye_distance": 0.0,
        }

    left_eye = pts[0]
    right_eye = pts[1]
    nose = pts[2]
    left_mouth = pts[3]
    right_mouth = pts[4]

    # Eye distance
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_dist = math.hypot(eye_dx, eye_dy)
    if eye_dist < 1e-4:
        return {"valid": False, "yaw_ratio": 1.0, "roll_deg": 90.0, "pitch_ratio": 0.0, "eye_distance": 0.0}

    # Roll angle (rotation in image plane)
    roll_deg = abs(math.degrees(math.atan2(eye_dy, eye_dx)))

    # Yaw ratio (horizontal turning)
    eye_mid = (left_eye + right_eye) / 2.0
    # Vector perpendicular to eyes
    eye_dir = np.array([eye_dx, eye_dy]) / eye_dist
    nose_vec = nose - eye_mid
    # Projection of nose offset along eye axis
    yaw_offset = np.dot(nose_vec, eye_dir)
    yaw_ratio = abs(yaw_offset / eye_dist)

    # Pitch ratio (vertical tilt)
    mouth_mid = (left_mouth + right_mouth) / 2.0
    # Distance from eye midpoint to nose, and nose to mouth midpoint
    eye_to_nose = np.linalg.norm(nose - eye_mid)
    nose_to_mouth = np.linalg.norm(mouth_mid - nose)
    
    pitch_ratio = eye_to_nose / max(nose_to_mouth, 1e-4)

    return {
        "valid": True,
        "yaw_ratio": float(yaw_ratio),
        "roll_deg": float(roll_deg),
        "pitch_ratio": float(pitch_ratio),
        "eye_distance": float(eye_dist),
    }


def filter_5_landmarks(
    landmarks: np.ndarray,
    face_bbox: Sequence[float] | np.ndarray,
    face_conf: float,
    cfg: LandmarkFilterConfig,
) -> Tuple[bool, str, dict]:
    """Evaluates if the 5 landmarks satisfy quality/pose criteria.
    
    Returns:
        (is_valid, reason, metrics)
    """
    if not cfg.enabled:
        return True, "Filter disabled", {}

    if face_conf < cfg.min_conf:
        return False, f"Face confidence {face_conf:.2f} < {cfg.min_conf:.2f}", {}

    # Check bounding box width/height
    w = face_bbox[2] - face_bbox[0]
    h = face_bbox[3] - face_bbox[1]
    if min(w, h) < cfg.min_face_size:
        return False, f"Face size {min(w, h):.1f}px < {cfg.min_face_size:.1f}px", {}

    metrics = calculate_landmark_geometry(landmarks)
    if not metrics["valid"]:
        return False, "Invalid landmark geometry", metrics

    if metrics["yaw_ratio"] > cfg.max_yaw_ratio:
        return False, f"Yaw ratio {metrics['yaw_ratio']:.2f} > {cfg.max_yaw_ratio:.2f}", metrics

    if metrics["roll_deg"] > cfg.max_roll_deg:
        return False, f"Roll angle {metrics['roll_deg']:.1f}° > {cfg.max_roll_deg:.1f}°", metrics

    if not (cfg.min_pitch_ratio <= metrics["pitch_ratio"] <= cfg.max_pitch_ratio):
        return False, f"Pitch ratio {metrics['pitch_ratio']:.2f} out of [{cfg.min_pitch_ratio:.2f}, {cfg.max_pitch_ratio:.2f}]", metrics

    return True, "Passed", metrics


def estimate_laplacian_sharpness(crop: np.ndarray) -> float:
    """Computes normalized Laplacian variance as a blur/sharpness score."""
    if crop.size == 0:
        return 0.0
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize with sigmoid-like curve
    return float(np.tanh(var / 300.0))


class FaceQualityEstimator:
    """Wraps eDifFIQA-T quality model with fallback."""

    def __init__(self, cfg: FaceQualityConfig, providers: Optional[list[str]] = None) -> None:
        self.cfg = cfg
        self.providers = providers
        self.model = None
        self._last_warn_time: float = 0.0
        if EDifFIQA is not None:
            try:
                self.model = EDifFIQA(providers=providers)
            except Exception:
                try:
                    self.model = EDifFIQA(providers=["CPUExecutionProvider"])
                except Exception as e:
                    print(f"[FaceQualityEstimator] Warning: Could not initialize EDifFIQA: {e}")

    def score(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        """Evaluates face quality score in range [0.0, 1.0]."""
        if not self.cfg.enabled:
            return 1.0

        if self.model is not None:
            try:
                res = self.model.predict(frame, landmarks)
                if hasattr(res, "score"):
                    return float(res.score)
            except Exception as e:
                now = time.time()
                if now - self._last_warn_time >= 60.0:
                    logger.warning(f"[FaceQualityEstimator] eDifFIQA predict failed: {e}")
                    self._last_warn_time = now
                return 0.0

        # Fallback heuristic based on sharpness and geometry (only when eDifFIQA model is not loaded)
        geo = calculate_landmark_geometry(landmarks)
        pts = np.asarray(landmarks, dtype=np.float32)
        x1, y1 = np.min(pts, axis=0).astype(int)
        x2, y2 = np.max(pts, axis=0).astype(int)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        # Resize to standard size before Laplacian to remove size bias
        if crop.size > 0:
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        sharpness = estimate_laplacian_sharpness(crop)
        pose_penalty = min(1.0, max(0.0, 1.0 - (geo.get("yaw_ratio", 0.0) + geo.get("roll_deg", 0.0) / 45.0)))
        return float(0.5 * sharpness + 0.5 * pose_penalty)

    def score_aligned(self, aligned_face: np.ndarray) -> float:
        """Scores an already aligned 112x112 face crop."""
        if not self.cfg.enabled:
            return 1.0
        if self.model is not None:
            try:
                res = self.model.score_aligned(aligned_face)
                if hasattr(res, "score"):
                    return float(res.score)
            except Exception as e:
                now = time.time()
                if now - self._last_warn_time >= 60.0:
                    logger.warning(f"[FaceQualityEstimator] eDifFIQA score_aligned failed: {e}")
                    self._last_warn_time = now
                return 0.0
        return estimate_laplacian_sharpness(aligned_face)


def score_face_quality(estimator: FaceQualityEstimator, frame: np.ndarray, landmarks: np.ndarray) -> float:
    return estimator.score(frame, landmarks)
