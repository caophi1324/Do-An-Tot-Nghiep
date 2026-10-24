"""Fake objects and test doubles to test Pipeline and Service components without loading heavy ONNX models."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from core.sources import VideoSource
from core.detectors import PersonDetection, FaceDetectionResult
from core.config import AppConfig


class FakeSource(VideoSource):
    """Test video source feeding pre-configured sequence of (frame, timestamp) tuples."""

    def __init__(
        self,
        frames: Sequence[Tuple[np.ndarray, float]],
        source_id: str = "fake_source",
        loop: bool = False,
    ) -> None:
        self.frames = list(frames)
        self.loop = loop
        self.idx = 0
        self._connected = True
        self.source_id = source_id

    def read(self) -> Tuple[float, Optional[np.ndarray]]:
        if not self._connected or not self.frames:
            return 0.0, None

        if self.idx >= len(self.frames):
            if self.loop:
                self.idx = 0
            else:
                return 0.0, None

        frame, ts = self.frames[self.idx]
        self.idx += 1
        return ts, frame.copy()

    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._connected = False

    def get_info(self) -> dict:
        w, h = (100, 100)
        if self.frames and self.frames[0][0] is not None:
            h, w = self.frames[0][0].shape[:2]
        return {
            "source_type": "fake",
            "source_id": self.source_id,
            "width": w,
            "height": h,
            "fps": 30.0,
        }

    def release(self) -> None:
        self._connected = False

    def reconnect(self) -> bool:
        self.idx = 0
        self._connected = True
        return True


class FakePersonDetector:
    """Mock person detector returning predefined bounding boxes."""

    def __init__(self, detections: Optional[List[PersonDetection]] = None) -> None:
        self.detections = detections if detections is not None else []
        self.call_count = 0

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        self.call_count += 1
        return list(self.detections)


class FakeFaceDetector:
    """Mock face detector returning predefined landmarks and bounding boxes."""

    def __init__(
        self,
        face_result: Optional[FaceDetectionResult] = None,
        frame_faces: Optional[List[FaceDetectionResult]] = None,
    ) -> None:
        self.face_result = face_result
        self.frame_faces = frame_faces if frame_faces is not None else []
        self.crop_call_count = 0
        self.frame_call_count = 0

    def detect_in_person_crop(self, frame: np.ndarray, person_bbox: np.ndarray) -> Optional[FaceDetectionResult]:
        self.crop_call_count += 1
        return self.face_result

    def detect_in_frame(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        self.frame_call_count += 1
        return list(self.frame_faces)


class FakeQualityEstimator:
    """Mock quality estimator returning fixed or configured quality scores."""

    def __init__(self, score_value: float = 0.85) -> None:
        self.score_value = score_value

    def score(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        return self.score_value

    def score_aligned(self, aligned_face: np.ndarray) -> float:
        return self.score_value


class FakeRecognizer:
    """Mock face recognizer with dictionary-backed template search and deterministic embeddings."""

    def __init__(self, templates: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
        self.templates = dict(templates) if templates else {}  # id -> {emb, meta}
        self.embeddings_dict: Dict[int, np.ndarray] = {}
        self.next_id = 1
        self._lock_count = 0
        self.active_person_ids: Optional[set[int]] = None

    def set_active_persons(self, active_ids: Optional[Sequence[int] | set[int]]) -> None:
        self.active_person_ids = set(active_ids) if active_ids is not None else None

    def is_person_valid(self, person_id: Optional[int]) -> bool:
        if person_id is None:
            return False
        if self.active_person_ids is None:
            return True
        return person_id in self.active_person_ids

    @property
    def total_templates(self) -> int:
        return len(self.templates)

    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aligned = np.full((112, 112, 3), 128, dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float32)
        return aligned, M

    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        # Deterministic unit vector
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        return v

    def search(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[Dict[str, Any]], float]:
        norm_e = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_meta = None
        best_sim = -1.0
        for tid, data in self.templates.items():
            t_emb = data["emb"]
            sim = float(np.dot(norm_e, t_emb))
            if sim > best_sim:
                best_sim = sim
                best_meta = data["meta"]
        if best_sim >= threshold and best_meta is not None:
            pid = best_meta.get("person_id")
            if self.active_person_ids is not None and pid not in self.active_person_ids:
                return None, 0.0
            return best_meta, best_sim
        return None, max(0.0, best_sim)

    def add_template(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        tid = self.next_id
        self.next_id += 1
        norm_e = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.templates[tid] = {"emb": norm_e, "meta": metadata}
        return tid

    def evaluate_distinctiveness(self, embedding: np.ndarray, target_person_id: Optional[int] = None) -> Dict[str, Any]:
        return {
            "is_distinct": True,
            "is_redundant": False,
            "is_mismatch": False,
            "sim_self": 0.90,
            "sim_other": 0.40,
            "margin": 0.50,
            "closest_other_id": None,
            "reason": "distinct",
        }

    def reset(self) -> None:
        self.templates.clear()
        self.next_id = 1

    def save(self) -> None:
        pass


def build_fake_pipeline(
    monkeypatch,
    cfg: Optional[AppConfig] = None,
    person_detections: Optional[List[PersonDetection]] = None,
    face_result: Optional[FaceDetectionResult] = None,
    quality_score: float = 0.85,
    on_template_learned: Optional[Callable] = None,
):
    """Constructs a real Pipeline instance but patches detector classes so no ONNX models are loaded."""
    from core.pipeline import Pipeline

    if cfg is None:
        cfg = AppConfig()

    fake_pdet = FakePersonDetector(person_detections)
    fake_fdet = FakeFaceDetector(face_result=face_result)
    fake_q = FakeQualityEstimator(score_value=quality_score)
    fake_rec = FakeRecognizer()

    monkeypatch.setattr("core.pipeline.PersonDetector", lambda *args, **kwargs: fake_pdet)
    monkeypatch.setattr("core.pipeline.FaceDetector", lambda *args, **kwargs: fake_fdet)
    monkeypatch.setattr("core.pipeline.FaceQualityEstimator", lambda *args, **kwargs: fake_q)
    monkeypatch.setattr("core.pipeline.FaceRecognizer", lambda *args, **kwargs: fake_rec)

    pipeline = Pipeline(cfg, on_template_learned=on_template_learned)
    return pipeline, fake_pdet, fake_fdet, fake_q, fake_rec
