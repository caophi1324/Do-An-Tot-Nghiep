"""TrackState and TrackManager wrapping ByteTrack with person/face tracking modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np

from uniface.tracking import BYTETracker
from .config import TrackingConfig, TrackingMode, RecognitionConfig, SmoothingConfig
from .smoothing import BoxSmoother


@dataclass
class TrackResult:
    track_id: int
    raw_bbox: np.ndarray  # [x1, y1, x2, y2]
    smoothed_bbox: np.ndarray  # [x1, y1, x2, y2]
    person_id: Optional[int] = None
    person_name: str = "Unknown"
    similarity: float = 0.0
    quality: float = 0.0
    is_unknown: bool = True
    face_bbox: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    lost_frames: int = 0
    timestamp: float = 0.0
    is_locked: bool = False
    learning_count: int = 1
    learning_target: int = 5
    learning_margin: float = 0.0
    is_learning_complete: bool = False


class FaceGallery:
    """Manages multi-angle face templates (quota=5), 2-stage staging buffer, and diversity checks."""

    def __init__(self, target_quota: int = 5) -> None:
        self.target_quota = target_quota
        self.anchor_embedding: Optional[np.ndarray] = None
        self.anchor_crop: Optional[np.ndarray] = None
        self.items: List[Tuple[np.ndarray, float, Optional[np.ndarray]]] = []  # (emb, quality, crop)
        self.staging_emb: Optional[np.ndarray] = None
        self.staging_quality: float = 0.0
        self.staging_crop: Optional[np.ndarray] = None
        self.staging_time: float = 0.0
        self.last_margin: float = 0.0
        self.learn_block_until: float = 0.0

    @property
    def total_count(self) -> int:
        return (1 if self.anchor_embedding is not None else 0) + len(self.items)

    @property
    def is_complete(self) -> bool:
        return self.total_count >= self.target_quota

    def set_anchor(self, emb: np.ndarray, quality: float, crop: Optional[np.ndarray] = None) -> None:
        norm = emb / max(1e-6, np.linalg.norm(emb))
        self.anchor_embedding = norm
        self.anchor_crop = crop

    def propose_candidate(
        self,
        emb: np.ndarray,
        quality: float,
        crop: Optional[np.ndarray],
        eval_result: Dict[str, Any],
        now: float,
    ) -> Optional[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
        """Proposes a new candidate face sample with 2-stage staging buffer and margin confirmation."""
        if self.is_complete or now < self.learn_block_until:
            return None

        if not eval_result.get("is_distinct", False):
            # Only wipe staging buffer on explicit mismatch or ambiguity, not mere redundancy
            if eval_result.get("is_mismatch", False) or eval_result.get("is_ambiguous", False):
                self.staging_emb = None
            return None

        norm_emb = emb / max(1e-6, np.linalg.norm(emb))
        self.last_margin = float(eval_result.get("margin", 0.0))

        # Staging Step 1: Buffer candidate
        if self.staging_emb is None:
            self.staging_emb = norm_emb
            self.staging_quality = quality
            self.staging_crop = crop
            self.staging_time = now
            return None

        # Staging Step 2: Confirm with consecutive frame (pair_sim >= 0.75)
        pair_sim = float(np.dot(self.staging_emb, norm_emb))
        if pair_sim >= 0.75:
            best_quality = max(quality, self.staging_quality)
            best_crop = crop if quality >= self.staging_quality else self.staging_crop
            committed = (norm_emb, best_quality, best_crop)
            self.items.append(committed)
            self.staging_emb = None
            self.learn_block_until = now + 0.3
            return committed
        else:
            self.staging_emb = norm_emb
            self.staging_quality = quality
            self.staging_crop = crop
            self.staging_time = now
            return None


class TrackState:
    """Maintains state, smoothing, best face templates, and identity voting per track."""

    def __init__(
        self,
        track_id: int,
        recog_cfg: RecognitionConfig,
        smooth_cfg: SmoothingConfig,
    ) -> None:
        self.track_id = track_id
        self.recog_cfg = recog_cfg
        self.smooth_cfg = smooth_cfg

        self.smoother = BoxSmoother(
            min_cutoff=smooth_cfg.min_cutoff,
            beta=smooth_cfg.beta,
            d_cutoff=smooth_cfg.d_cutoff,
        )
        self.face_smoother = BoxSmoother(
            min_cutoff=smooth_cfg.min_cutoff,
            beta=smooth_cfg.beta,
            d_cutoff=smooth_cfg.d_cutoff,
        )

        self.gallery = FaceGallery(target_quota=5)
        self.votes: List[Tuple[Optional[int], str, float]] = []  # (person_id, person_name, sim)
        self.person_id: Optional[int] = None
        self.person_name: str = "Unknown"
        self.similarity: float = 0.0
        self.is_unknown: bool = True

        self.best_quality: float = 0.0
        self.best_embedding: Optional[np.ndarray] = None
        self.best_face_crop: Optional[np.ndarray] = None
        self.best_landmarks: Optional[np.ndarray] = None

        self.last_seen_time: float = 0.0
        self.last_recognition_time: float = 0.0
        self.lost_frames: int = 0

        self.last_face_box: Optional[np.ndarray] = None
        self.last_landmarks: Optional[np.ndarray] = None
        self.relative_face_box: Optional[Tuple[float, float, float, float]] = None
        self.face_eval_count: int = 0
        self.consecutive_landmark_misses: int = 0

    def update_config(self, recog_cfg: RecognitionConfig, smooth_cfg: SmoothingConfig) -> None:
        """Dynamically applies updated recognition and smoothing configuration."""
        self.recog_cfg = recog_cfg
        self.smooth_cfg = smooth_cfg
        self.smoother.update_params(
            min_cutoff=smooth_cfg.min_cutoff,
            beta=smooth_cfg.beta,
            d_cutoff=smooth_cfg.d_cutoff,
        )
        self.face_smoother.update_params(
            min_cutoff=smooth_cfg.min_cutoff,
            beta=smooth_cfg.beta,
            d_cutoff=smooth_cfg.d_cutoff,
        )

    def vote(self, pid: Optional[int], name: str, sim: float) -> None:
        """Adds a recognition observation and evaluates voting consensus."""
        self.votes.append((pid, name, sim))
        if len(self.votes) > self.recog_cfg.vote_window:
            self.votes.pop(0)

        # Count known valid votes
        valid_votes = [(p, n, s) for p, n, s in self.votes if p is not None]
        if not valid_votes:
            # If all recent votes are Unknown, drop identity back to Unknown
            if len(self.votes) >= self.recog_cfg.vote_min:
                self.person_id = None
                self.person_name = "Unknown"
                self.is_unknown = True
                self.similarity = float(np.mean([s for _, _, s in self.votes])) if self.votes else 0.0
            return

        # Tally person IDs
        counts: Dict[int, int] = {}
        names_map: Dict[int, str] = {}
        sims_map: Dict[int, List[float]] = {}
        for p, n, s in valid_votes:
            counts[p] = counts.get(p, 0) + 1
            names_map[p] = n
            sims_map.setdefault(p, []).append(s)

        top_pid = max(counts, key=counts.get)
        top_count = counts[top_pid]
        avg_sim = float(np.mean(sims_map[top_pid]))

        if self.person_id is None:
            # First confirmation: requires >= vote_min identical votes and similarity >= hysteresis_thr
            if top_count >= self.recog_cfg.vote_min and avg_sim >= self.recog_cfg.hysteresis_thr:
                self.person_id = top_pid
                self.person_name = names_map[top_pid]
                self.similarity = avg_sim
                self.is_unknown = False
        else:
            cur_count = counts.get(self.person_id, 0)
            cur_sim = float(np.mean(sims_map[self.person_id])) if cur_count > 0 else 0.0
            non_cur_votes = len(self.votes) - cur_count

            # Hysteresis consensus evaluation:
            # 1. Switch identity if a different ID has overwhelming consensus
            if top_pid != self.person_id and top_count >= self.recog_cfg.vote_min and top_count > cur_count + 1 and avg_sim >= self.recog_cfg.hysteresis_thr:
                self.person_id = top_pid
                self.person_name = names_map[top_pid]
                self.similarity = avg_sim
                self.is_unknown = False
            # 2. Retain identity if current identity is supported and meets hysteresis threshold
            elif cur_count > 0 and cur_sim >= self.recog_cfg.hysteresis_thr and non_cur_votes < self.recog_cfg.vote_min:
                self.similarity = cur_sim
            # 3. Drop to Unknown if Unknown/competing votes reach vote_min or similarity drops below hysteresis threshold
            elif non_cur_votes >= self.recog_cfg.vote_min or cur_count == 0 or cur_sim < self.recog_cfg.hysteresis_thr:
                old_pid = self.person_id
                self.person_id = None
                self.person_name = "Unknown"
                self.is_unknown = True
                self.similarity = cur_sim if cur_count > 0 else 0.0
                if cur_sim < self.recog_cfg.hysteresis_thr:
                    self.votes = [(p, n, s) for p, n, s in self.votes if p != old_pid]
            else:
                self.similarity = cur_sim

    def smooth_bbox(self, box: Sequence[float] | np.ndarray, timestamp: float) -> np.ndarray:
        if not self.smooth_cfg.enabled:
            return np.asarray(box, dtype=np.float32)
        return self.smoother.update(box, timestamp)

    def smooth_face_bbox(self, box: Sequence[float] | np.ndarray | None, timestamp: float) -> Optional[np.ndarray]:
        if box is None:
            return None
        if not self.smooth_cfg.enabled:
            return np.asarray(box, dtype=np.float32)
        return self.face_smoother.update(box, timestamp)


class TrackManager:
    """Manages ByteTrack instance and per-track states."""

    def __init__(
        self,
        track_cfg: TrackingConfig,
        recog_cfg: RecognitionConfig,
        smooth_cfg: SmoothingConfig,
    ) -> None:
        self.track_cfg = track_cfg
        self.recog_cfg = recog_cfg
        self.smooth_cfg = smooth_cfg

        self.tracker = BYTETracker(
            track_thresh=track_cfg.track_thresh,
            track_buffer=track_cfg.track_buffer,
            match_thresh=track_cfg.match_thresh,
            low_thresh=track_cfg.low_thresh,
        )
        self.states: Dict[int, TrackState] = {}

    def update_config(
        self,
        track_cfg: TrackingConfig,
        recog_cfg: RecognitionConfig,
        smooth_cfg: SmoothingConfig,
    ) -> None:
        """Dynamically applies tracking, recognition, and smoothing configurations."""
        need_rebuild = (
            self.track_cfg.track_thresh != track_cfg.track_thresh
            or self.track_cfg.track_buffer != track_cfg.track_buffer
            or self.track_cfg.match_thresh != track_cfg.match_thresh
            or self.track_cfg.low_thresh != track_cfg.low_thresh
        )
        self.track_cfg = track_cfg
        self.recog_cfg = recog_cfg
        self.smooth_cfg = smooth_cfg

        if need_rebuild:
            self.tracker = BYTETracker(
                track_thresh=track_cfg.track_thresh,
                track_buffer=track_cfg.track_buffer,
                match_thresh=track_cfg.match_thresh,
                low_thresh=track_cfg.low_thresh,
            )

        for state in self.states.values():
            state.update_config(recog_cfg, smooth_cfg)

    def reset(self) -> None:
        self.tracker.reset()
        self.states.clear()

    def update(
        self,
        detections_xyxy_conf: np.ndarray,
        timestamp: float,
    ) -> List[Tuple[int, np.ndarray, TrackState]]:
        """Updates tracker with detections [x1, y1, x2, y2, score].
        
        Returns:
            List of (track_id, raw_bbox, track_state)
        """
        if detections_xyxy_conf is None or len(detections_xyxy_conf) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(detections_xyxy_conf, dtype=np.float32)

        # ByteTrack update returns [x1, y1, x2, y2, track_id] or [x1, y1, x2, y2, track_id, score]
        active_tracks = self.tracker.update(dets)

        results = []
        active_ids = set()

        if active_tracks is not None and len(active_tracks) > 0:
            for trk in active_tracks:
                x1, y1, x2, y2 = float(trk[0]), float(trk[1]), float(trk[2]), float(trk[3])
                tid = int(trk[4])
                active_ids.add(tid)

                if tid not in self.states:
                    self.states[tid] = TrackState(
                        track_id=tid,
                        recog_cfg=self.recog_cfg,
                        smooth_cfg=self.smooth_cfg,
                    )

                state = self.states[tid]
                state.last_seen_time = timestamp
                state.lost_frames = 0
                raw_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                results.append((tid, raw_box, state))

        # Cleanup disappeared tracks
        stale_tids = [tid for tid in self.states if tid not in active_ids]
        for tid in stale_tids:
            self.states[tid].lost_frames += 1
            if self.states[tid].lost_frames > self.track_cfg.track_buffer:
                del self.states[tid]
            elif (
                self.smooth_cfg.hold_lost_frames > 0
                and self.states[tid].lost_frames <= self.smooth_cfg.hold_lost_frames
                and self.states[tid].smoother.last_box is not None
            ):
                results.append((tid, self.states[tid].smoother.last_box, self.states[tid]))

        return results
