"""Face alignment, EdgeFace embedding extraction, and FAISS vector matching."""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from uniface.recognition import EdgeFace
from uniface.face_utils import face_alignment
from uniface.stores import FAISS

from .config import RecognitionConfig


def align_face_112(image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Warps 5 facial landmarks into a normalized 112x112 face crop."""
    pts = np.asarray(landmarks, dtype=np.float32)
    aligned, transform_matrix = face_alignment(image, pts, image_size=112)
    return aligned, transform_matrix


try:
    from uniface.constants import EdgeFaceWeights
except ImportError:
    EdgeFaceWeights = None


def get_edgeface_weight(name: Any) -> Any:
    if EdgeFaceWeights is None or isinstance(name, EdgeFaceWeights):
        return name
    name_str = str(name).lower()
    for member in EdgeFaceWeights:
        if member.value.lower() == name_str or member.name.lower() == name_str:
            return member
    return EdgeFaceWeights.XXS


class FaceRecognizer:
    """Manages EdgeFace model inference and FAISS search index."""

    def __init__(
        self,
        cfg: RecognitionConfig,
        faiss_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        if faiss_path is None:
            faiss_path = os.environ.get("FAISS_PATH", "data/faiss_index")
        self.faiss_path = faiss_path
        self.providers = providers
        self.active_person_ids: Optional[set[int]] = None
        model_name = get_edgeface_weight(getattr(cfg, "model_name", "edgeface_xxs"))
        try:
            self.model = EdgeFace(model_name=model_name, providers=providers)
        except Exception as e_primary:
            try:
                self.model = EdgeFace(model_name=model_name, providers=["CPUExecutionProvider"])
            except Exception as e_fallback:
                raise RuntimeError(
                    f"[FaceRecognizer] Failed to initialize EdgeFace with providers {providers} ({e_primary}) "
                    f"and CPU fallback failed ({e_fallback})"
                ) from e_fallback
        self.store = FAISS(embedding_size=512, db_path=faiss_path)
        self._faiss_lock = threading.RLock()
        self._init_store()

    def set_active_persons(self, active_ids: Optional[Sequence[int] | set[int]]) -> None:
        """Sets the set of active (non-soft-deleted) person IDs."""
        self.active_person_ids = set(active_ids) if active_ids is not None else None

    def is_person_valid(self, person_id: Optional[int]) -> bool:
        """Checks if a person_id is active (not soft-deleted)."""
        if person_id is None:
            return False
        if self.active_person_ids is None:
            return True
        return person_id in self.active_person_ids

    def _init_store(self) -> None:
        os.makedirs(self.faiss_path, exist_ok=True)
        try:
            self.store.load()
        except Exception:
            # First initialization or empty store
            pass

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align face to 112x112."""
        return align_face_112(image, landmarks)

    def extract_embedding(self, image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts 512-dim L2-normalized embedding.
        
        If landmarks are provided, performs alignment first.
        If image is already aligned 112x112, landmarks can be None.
        """
        if landmarks is not None:
            emb = self.model.get_normalized_embedding(image, np.asarray(landmarks, dtype=np.float32))
        else:
            emb = self.model.get_embedding(image)
            # Ensure L2 normalization
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb

    def search(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """Searches FAISS vector store for closest identity (thread-safe)."""
        if not self.cfg.enabled or self.store.size == 0:
            return None, 0.0

        thr = threshold if threshold is not None else self.cfg.match_thr
        with self._faiss_lock:
            try:
                metadata, sim = self.store.search(embedding, threshold=thr)
                if metadata is not None:
                    pid = metadata.get("person_id")
                    if self.active_person_ids is not None and pid not in self.active_person_ids:
                        # Soft-deleted person: ignore match
                        return None, 0.0
                return metadata, float(sim)
            except Exception as e:
                print(f"[FaceRecognizer] Search error: {e}")
                return None, 0.0

    def add_template(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Adds template embedding to FAISS vector index (thread-safe)."""
        with self._faiss_lock:
            self.store.add(embedding, metadata)

    def save(self) -> None:
        """Persists FAISS index to disk."""
        with self._faiss_lock:
            os.makedirs(self.faiss_path, exist_ok=True)
            self.store.save()

    def rebuild_index(self, templates: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Rebuilds the FAISS store from a list of (embedding, metadata) (thread-safe)."""
        new_store = FAISS(embedding_size=512, db_path=self.faiss_path)
        for emb, meta in templates:
            new_store.add(emb, meta)
        with self._faiss_lock:
            self.store = new_store
            self.store.save()

    @property
    def total_templates(self) -> int:
        return int(self.store.size)

    def evaluate_distinctiveness(
        self,
        embedding: np.ndarray,
        target_person_id: Optional[int],
        margin_thr: float = 0.15,
        min_self_sim: float = 0.50,
        max_self_sim: float = 0.99,
    ) -> Dict[str, Any]:
        """Evaluates whether an embedding is distinctively matching target_person_id while clearly separated from other IDs."""
        with self._faiss_lock:
            if self.store.size == 0 or self.store.index.ntotal == 0:
                return {
                    "is_distinct": True,
                    "is_redundant": False,
                    "is_mismatch": False,
                    "is_ambiguous": False,
                    "sim_self": 1.0,
                    "sim_other": 0.0,
                    "margin": 1.0,
                    "closest_other_id": None,
                    "reason": "empty_database",
                }

            vec = self.store._prepare(embedding).reshape(1, -1)
            k = min(self.store.index.ntotal, 50)
            similarities, indices = self.store.index.search(vec, k)

            sim_self = 0.0
            sim_other = 0.0
            closest_other_id = None

            for s, idx in zip(similarities[0], indices[0]):
                if idx < 0 or idx >= len(self.store.metadata):
                    continue
                meta = self.store.metadata[idx]
                pid = meta.get("person_id")
                # Skip soft-deleted persons if active_person_ids is populated
                if self.active_person_ids is not None and pid not in self.active_person_ids:
                    continue
                sim_val = float(s)

                if target_person_id is not None and pid == target_person_id:
                    if sim_val > sim_self:
                        sim_self = sim_val
                else:
                    if sim_val > sim_other:
                        sim_other = sim_val
                        closest_other_id = pid

            margin = sim_self - sim_other

            # Analyze suitability
            is_redundant = (sim_self > max_self_sim)
            is_mismatch = (target_person_id is not None and sim_self < min_self_sim)
            # Ambiguous if another person matches closely and margin is small,
            # UNLESS self-match is decisively high (sim_self >= 0.80, where a competing high match indicates a duplicate DB identity)
            is_ambiguous = (
                closest_other_id is not None
                and margin < margin_thr
                and sim_other >= 0.55
                and sim_self < 0.80
            )

            is_distinct = (
                not is_mismatch
                and not is_redundant
                and not is_ambiguous
            )

            if is_redundant:
                reason = f"redundant (sim_self={sim_self:.2f} > {max_self_sim:.2f})"
            elif is_mismatch:
                reason = f"mismatch (sim_self={sim_self:.2f} < {min_self_sim:.2f})"
            elif is_ambiguous:
                reason = f"ambiguous (margin={margin:+.2f} < {margin_thr:.2f}, close to ID {closest_other_id})"
            else:
                reason = f"distinct (sim_self={sim_self:.2f}, margin={margin:+.2f})"

            return {
                "is_distinct": is_distinct,
                "is_redundant": is_redundant,
                "is_mismatch": is_mismatch,
                "is_ambiguous": is_ambiguous,
                "sim_self": sim_self,
                "sim_other": sim_other,
                "margin": margin,
                "closest_other_id": closest_other_id,
                "reason": reason,
            }

    def reset(self) -> None:
        """Clears all vectors from FAISS index and resets storage."""
        with self._faiss_lock:
            self.store = FAISS(embedding_size=512, db_path=self.faiss_path)
            self.store.save()
