"""Coordinated Pipeline executing detection, tracking, filtering, quality, recognition, smoothing, and rendering."""
from __future__ import annotations

import time
import threading
from typing import Callable, Dict, List, Optional, Tuple, Sequence
import numpy as np
import cv2

from .config import AppConfig, TrackingMode, OverlayConfig
from .detectors import PersonDetector, FaceDetector, PersonDetection, FaceDetectionResult
from .quality import filter_5_landmarks, FaceQualityEstimator
from .tracker import TrackManager, TrackResult, TrackState
from .recognizer import FaceRecognizer


def propagate_landmarks_motion(
    old_landmarks: np.ndarray,
    old_box: np.ndarray,
    new_box: np.ndarray,
) -> np.ndarray:
    """Propagates 5-point landmarks dynamically according to bounding box displacement and scale change."""
    pts = np.asarray(old_landmarks, dtype=np.float32)
    old_w = max(1.0, float(old_box[2] - old_box[0]))
    old_h = max(1.0, float(old_box[3] - old_box[1]))
    old_cx = float(old_box[0] + old_box[2]) / 2.0
    old_cy = float(old_box[1] + old_box[3]) / 2.0

    new_w = max(1.0, float(new_box[2] - new_box[0]))
    new_h = max(1.0, float(new_box[3] - new_box[1]))
    new_cx = float(new_box[0] + new_box[2]) / 2.0
    new_cy = float(new_box[1] + new_box[3]) / 2.0

    scale_x = new_w / old_w
    scale_y = new_h / old_h

    new_landmarks = np.empty_like(pts)
    new_landmarks[:, 0] = new_cx + (pts[:, 0] - old_cx) * scale_x
    new_landmarks[:, 1] = new_cy + (pts[:, 1] - old_cy) * scale_y
    return new_landmarks


class Pipeline:
    """Master Pipeline orchestrating the end-to-end face recognition & tracking system."""

    def __init__(
        self,
        cfg: AppConfig,
        on_template_learned: Optional[Callable[[int, np.ndarray, float, Optional[np.ndarray]], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.on_template_learned = on_template_learned
        from .config import get_optimal_providers
        dev_id = getattr(cfg.hardware, "device_id", 0)
        providers = get_optimal_providers(cfg.hardware.provider)

        if dev_id > 0 and any("Dml" in str(p) for p in providers):
            uniface_providers = [
                ("DmlExecutionProvider", {"device_id": str(dev_id)}) if ("Dml" in str(p)) else p
                for p in providers
            ]
        else:
            uniface_providers = providers

        self.person_det = PersonDetector(cfg.person_det, providers=providers)
        self.face_det = FaceDetector(cfg.face_det, providers=uniface_providers)
        self.quality_estimator = FaceQualityEstimator(cfg.quality, providers=uniface_providers)
        self.recognizer = FaceRecognizer(cfg.recognition, faiss_path=cfg.storage.faiss_path, providers=uniface_providers)
        self.tracker = TrackManager(cfg.tracking, cfg.recognition, cfg.smoothing)
        self.frame_index = 0
        self.locked_track_id: Optional[int] = None
        self.locked_person_id: Optional[int] = None
        self._last_locked_box: Optional[np.ndarray] = None
        self._locked_lost_frames: int = 0
        self._active_mode: TrackingMode = cfg.tracking.mode
        self._inference_lock = threading.RLock()

    def lock_target(self, track_id: int, person_id: Optional[int] = None) -> None:
        """Locks onto a specific target by track_id and/or person_id."""
        with self._inference_lock:
            self.locked_track_id = track_id
            self.locked_person_id = person_id
            self._locked_lost_frames = 0
            state = self.tracker.states.get(track_id)
            if state is not None and state.last_face_box is not None:
                self._last_locked_box = state.last_face_box.copy()

    def unlock_target(self) -> None:
        """Clears target lock."""
        with self._inference_lock:
            self.locked_track_id = None
            self.locked_person_id = None
            self._last_locked_box = None
            self._locked_lost_frames = 0

    def update_config(self, new_cfg: AppConfig) -> None:
        """Dynamically applies updated configuration in real-time."""
        with self._inference_lock:
            mode_changed = (new_cfg.tracking.mode != self._active_mode)
            self._active_mode = new_cfg.tracking.mode
            self.cfg = new_cfg
            self.person_det.update_config(new_cfg.person_det)
            self.face_det.update_config(new_cfg.face_det)
            self.quality_estimator.cfg = new_cfg.quality
            self.recognizer.cfg = new_cfg.recognition
            self.tracker.update_config(new_cfg.tracking, new_cfg.recognition, new_cfg.smoothing)

            if mode_changed:
                self.tracker.reset()
                self.locked_track_id = None
                self.locked_person_id = None
                self._last_locked_box = None
                self._locked_lost_frames = 0

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> List[TrackResult]:
        """Processes a single video frame with thread-safe inference lock."""
        if frame is None or frame.size == 0:
            return []

        if timestamp is None:
            timestamp = time.time()

        with self._inference_lock:
            self.frame_index += 1
            if self.cfg.tracking.mode != self._active_mode:
                self._active_mode = self.cfg.tracking.mode
                self.tracker.reset()
                self.locked_track_id = None
                self.locked_person_id = None
                self._last_locked_box = None
                self._locked_lost_frames = 0

            if self.cfg.tracking.mode == TrackingMode.PERSON:
                return self._process_person_tracking_mode(frame, timestamp)
            else:
                return self._process_face_tracking_mode(frame, timestamp)

    def _enforce_exclusive_identities_and_locking(self, results: List[TrackResult]) -> List[TrackResult]:
        """Enforces exclusive 1-to-1 identity mapping and single-target locking with spatial continuity."""
        # 1. Enforce 1-to-1 Identity Constraint: No duplicate person_id in the same frame
        assigned_identities: Dict[int, TrackResult] = {}
        for r in sorted(results, key=lambda x: (x.track_id == self.locked_track_id, x.similarity), reverse=True):
            if r.person_id is not None and not r.is_unknown:
                # Invalidate if person has been soft-deleted
                if not self.recognizer.is_person_valid(r.person_id):
                    demoted_pid = r.person_id
                    r.person_id = None
                    r.person_name = "Unknown"
                    r.is_unknown = True
                    r.similarity = 0.0
                    state = self.tracker.states.get(r.track_id)
                    if state is not None:
                        state.person_id = None
                        state.person_name = "Unknown"
                        state.is_unknown = True
                        state.similarity = 0.0
                        state.votes = [(p, n, s) for p, n, s in state.votes if p != demoted_pid]
                    continue

                if r.person_id in assigned_identities:
                    # Duplicate identity detected! Demote the lower-priority duplicate to Unknown
                    demoted_pid = r.person_id
                    r.person_id = None
                    r.person_name = "Unknown"
                    r.is_unknown = True
                    r.similarity = 0.0
                    state = self.tracker.states.get(r.track_id)
                    if state is not None:
                        state.person_id = None
                        state.person_name = "Unknown"
                        state.is_unknown = True
                        state.similarity = 0.0
                        # Clear competing votes for demoted identity to prevent next-frame flicker
                        state.votes = [(p, n, s) for p, n, s in state.votes if p != demoted_pid]
                    continue

                assigned_identities[r.person_id] = r

        # 2. Strict Single-Target Locking with Spatial Continuity Guard
        effective_locked_tid = None
        active_tids = {r.track_id for r in results}

        if self.locked_track_id in active_tids:
            effective_locked_tid = self.locked_track_id
            self._locked_lost_frames = 0
            for r in results:
                if r.track_id == effective_locked_tid:
                    self._last_locked_box = r.smoothed_bbox.copy()
                    if r.person_id is not None and not r.is_unknown and self.locked_person_id is None:
                        self.locked_person_id = r.person_id
                    break
        else:
            # Locked track ID is not directly in active results!
            # Strategy A: Rebind via locked_person_id (if identity is known)
            if self.locked_person_id is not None:
                candidates = [r for r in results if r.person_id == self.locked_person_id and not r.is_unknown]
                if candidates:
                    best_cand = max(candidates, key=lambda c: c.similarity)
                    effective_locked_tid = best_cand.track_id
                    self.locked_track_id = effective_locked_tid
                    self._last_locked_box = best_cand.smoothed_bbox.copy()
                    self._locked_lost_frames = 0

            # Strategy B: Spatial Continuity Hand-off (IoU / spatial proximity with last locked box)
            if effective_locked_tid is None and self._last_locked_box is not None and self._locked_lost_frames < 15:
                best_iou = 0.0
                best_match_tid = None
                lx1, ly1, lx2, ly2 = self._last_locked_box[:4]
                l_area = max(1.0, float((lx2 - lx1) * (ly2 - ly1)))
                for r in results:
                    bx1, by1, bx2, by2 = r.smoothed_bbox[:4]
                    ix1, iy1 = max(lx1, bx1), max(ly1, by1)
                    ix2, iy2 = min(lx2, bx2), min(ly2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = float((ix2 - ix1) * (iy2 - iy1))
                        b_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
                        iou = inter / (l_area + b_area - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_tid = r.track_id

                if best_iou >= 0.30 and best_match_tid is not None:
                    effective_locked_tid = best_match_tid
                    self.locked_track_id = effective_locked_tid
                    self._locked_lost_frames = 0
                    for r in results:
                        if r.track_id == effective_locked_tid:
                            self._last_locked_box = r.smoothed_bbox.copy()
                            break

            if effective_locked_tid is None and (self.locked_track_id is not None or self.locked_person_id is not None):
                self._locked_lost_frames += 1
                if self._locked_lost_frames > 60:
                    self.locked_track_id = None
                    self.locked_person_id = None
                    self._last_locked_box = None
                    self._locked_lost_frames = 0

        for r in results:
            r.is_locked = (effective_locked_tid is not None and r.track_id == effective_locked_tid)

        return results


    def _process_person_tracking_mode(self, frame: np.ndarray, timestamp: float) -> List[TrackResult]:
        """People Detection -> ByteTrack -> Upper Crop Face Detection -> Landmark Filter -> Quality -> Recognition."""
        # 1. Detect People
        person_dets = self.person_det.detect(frame)
        if person_dets:
            dets_arr = np.array([[d.x1, d.y1, d.x2, d.y2, d.conf] for d in person_dets], dtype=np.float32)
        else:
            dets_arr = np.empty((0, 5), dtype=np.float32)

        # 2. Track People
        tracked = self.tracker.update(dets_arr, timestamp)
        results: List[TrackResult] = []

        for tid, raw_bbox, state in tracked:
            state.face_eval_count += 1
            # 1. Smooth Person Bounding Box
            smoothed_box = state.smooth_bbox(raw_bbox, timestamp)

            face_res: Optional[FaceDetectionResult] = None
            landmarks: Optional[np.ndarray] = None
            face_box: Optional[np.ndarray] = None

            # Staggered face detection cadence to save GPU/CPU cycles:
            # - Priority for locked target: check every frame if unconfirmed or learning, every 2 frames if confirmed
            # - Facing away (no face yet): test initial 2 frames, then every 3 frames
            # - If unconfirmed identity: run every 2nd frame (staggered by tid)
            # - If confirmed identity: run every 5th frame or when recheck interval elapses
            should_detect_face = False
            if tid == self.locked_track_id:
                if state.person_id is None or not state.gallery.is_complete:
                    should_detect_face = True
                else:
                    should_detect_face = ((state.face_eval_count + tid) % 2 == 0)
            elif state.last_face_box is None:
                if state.face_eval_count <= 2:
                    should_detect_face = True
                else:
                    should_detect_face = ((state.face_eval_count + tid) % 3 == 0)
            elif state.person_id is None:
                should_detect_face = ((state.face_eval_count + tid) % 2 == 0)
            else:
                time_since_recog = timestamp - state.last_recognition_time
                if time_since_recog >= self.cfg.recognition.recheck_interval_s or ((state.face_eval_count + tid) % 5 == 0):
                    should_detect_face = True

            if should_detect_face:
                # 3. Detect Face in upper crop of person bbox
                face_res = self.face_det.detect_in_person_crop(frame, raw_bbox)

            if face_res is not None and face_res.landmarks is not None:
                raw_fb = face_res.bbox
                raw_lm = face_res.landmarks

                # Update relative face position inside person bbox (using smoothed_box)
                pw = max(1.0, float(smoothed_box[2] - smoothed_box[0]))
                ph = max(1.0, float(smoothed_box[3] - smoothed_box[1]))
                state.relative_face_box = (
                    float(raw_fb[0] - smoothed_box[0]) / pw,
                    float(raw_fb[1] - smoothed_box[1]) / ph,
                    float(raw_fb[2] - smoothed_box[0]) / pw,
                    float(raw_fb[3] - smoothed_box[1]) / ph,
                )

                face_box = state.smooth_face_bbox(raw_fb, timestamp)
                landmarks = raw_lm
                state.last_face_box = face_box
                state.last_landmarks = landmarks
                state.consecutive_landmark_misses = 0

                # 4. Filter 5-point Landmarks (yaw, pitch, roll, size, conf)
                passed, reason, metrics = filter_5_landmarks(
                    landmarks=landmarks,
                    face_bbox=face_box,
                    face_conf=face_res.conf,
                    cfg=self.cfg.landmark_filter,
                )

                if passed:
                    # 5. Face Alignment (112x112) - align once
                    aligned, _ = self.recognizer.align(frame, landmarks)

                    # 6. Face Quality Assessment on aligned crop
                    if hasattr(self.quality_estimator, "score_aligned"):
                        q_score = self.quality_estimator.score_aligned(aligned)
                    else:
                        q_score = self.quality_estimator.score(frame, landmarks)
                    
                    # Decide if recognition should run (best face update, unconfirmed identity, or active learning target)
                    is_learning_target = (tid == self.locked_track_id and not state.gallery.is_complete)
                    should_recognize = False
                    if q_score >= self.cfg.quality.min_quality:
                        if state.person_id is None or is_learning_target:
                            should_recognize = True
                        elif q_score > (state.best_quality + self.cfg.quality.upgrade_delta):
                            should_recognize = True
                        elif (timestamp - state.last_recognition_time) >= self.cfg.recognition.recheck_interval_s:
                            should_recognize = True

                    if should_recognize:
                        state.best_quality = max(state.best_quality, q_score)
                        state.last_recognition_time = timestamp

                        # 7. Extract Embedding from already-aligned crop
                        emb = self.recognizer.extract_embedding(aligned)
                        state.best_embedding = emb
                        state.best_landmarks = landmarks
                        state.best_face_crop = aligned

                        # 8. Search FAISS vector database
                        match_meta, sim = self.recognizer.search(emb, threshold=self.cfg.recognition.match_thr)
                        if match_meta is not None:
                            pid = match_meta.get("person_id")
                            name = match_meta.get("full_name", match_meta.get("name", "Known"))
                            state.vote(pid=pid, name=name, sim=sim)
                        else:
                            state.vote(pid=None, name="Unknown", sim=sim)

                        # 8b. Active 5-Sample Learning & Cross-ID Distinctiveness
                        # Crowd guard check against other persons in scene
                        is_crowded = False
                        all_other_boxes = [box for other_tid, box, _ in tracked if other_tid != tid]
                        tx1, ty1, tx2, ty2 = raw_bbox[:4]
                        t_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
                        for ob in all_other_boxes:
                            ox1, oy1, ox2, oy2 = ob[:4]
                            ix1, iy1 = max(tx1, ox1), max(ty1, oy1)
                            ix2, iy2 = min(tx2, ox2), min(ty2, oy2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = float((ix2 - ix1) * (iy2 - iy1))
                                o_area = max(1.0, float((ox2 - ox1) * (oy2 - oy1)))
                                if (inter / (t_area + o_area - inter)) > 0.25:
                                    is_crowded = True
                                    break

                        target_pid = state.person_id or (self.locked_person_id if self.locked_track_id == tid else None)
                        if is_crowded:
                            state.gallery.learn_block_until = timestamp + 0.3
                        elif target_pid is not None and self.recognizer.is_person_valid(target_pid):
                            if state.gallery.anchor_embedding is None:
                                state.gallery.set_anchor(emb, q_score, aligned)
                            if not state.gallery.is_complete:
                                eval_res = self.recognizer.evaluate_distinctiveness(emb, target_person_id=target_pid)
                                committed = state.gallery.propose_candidate(emb, q_score, aligned, eval_res, timestamp)
                                if committed is not None:
                                    norm_emb, q, crop = committed
                                    self.recognizer.add_template(norm_emb, {"person_id": target_pid, "name": state.person_name})
                                    if self.on_template_learned:
                                        self.on_template_learned(target_pid, norm_emb, q, crop)
            else:
                # Cadence skip or transient miss: extrapolate smoothly from tracked person box
                state.consecutive_landmark_misses += 1
                if state.relative_face_box is not None and state.consecutive_landmark_misses <= 15:
                    pw = float(smoothed_box[2] - smoothed_box[0])
                    ph = float(smoothed_box[3] - smoothed_box[1])
                    rx1, ry1, rx2, ry2 = state.relative_face_box
                    extrapolated_box = np.array([
                        smoothed_box[0] + rx1 * pw,
                        smoothed_box[1] + ry1 * ph,
                        smoothed_box[0] + rx2 * pw,
                        smoothed_box[1] + ry2 * ph,
                    ], dtype=np.float32)
                    face_box = state.smooth_face_bbox(extrapolated_box, timestamp)

                    # Dynamically propagate 5 landmarks using displacement & scale
                    if state.last_landmarks is not None and state.last_face_box is not None:
                        landmarks = propagate_landmarks_motion(state.last_landmarks, state.last_face_box, face_box)
                        state.last_face_box = face_box
                        state.last_landmarks = landmarks
                    else:
                        landmarks = state.last_landmarks
                elif state.last_face_box is not None and state.consecutive_landmark_misses <= 10:
                    face_box = state.smooth_face_bbox(state.last_face_box, timestamp)
                    landmarks = state.last_landmarks
                else:
                    face_box = None
                    landmarks = None

            res = TrackResult(
                track_id=tid,
                raw_bbox=raw_bbox,
                smoothed_bbox=smoothed_box,
                person_id=state.person_id,
                person_name=state.person_name,
                similarity=state.similarity,
                quality=state.best_quality,
                is_unknown=state.is_unknown,
                face_bbox=face_box,
                landmarks=landmarks,
                lost_frames=state.lost_frames,
                timestamp=timestamp,
                is_locked=False,
                learning_count=state.gallery.total_count,
                learning_target=state.gallery.target_quota,
                learning_margin=state.gallery.last_margin,
                is_learning_complete=state.gallery.is_complete,
            )
            results.append(res)

        return self._enforce_exclusive_identities_and_locking(results)

    def _process_face_tracking_mode(self, frame: np.ndarray, timestamp: float) -> List[TrackResult]:
        """Direct Face Detection -> ByteTrack -> Landmark Filter -> Quality -> Recognition."""
        # 1. Detect Faces
        faces = self.face_det.detect_in_frame(frame)
        if faces:
            dets_arr = np.array([[f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.conf] for f in faces], dtype=np.float32)
        else:
            dets_arr = np.empty((0, 5), dtype=np.float32)

        # 2. Track Faces
        tracked = self.tracker.update(dets_arr, timestamp)
        results: List[TrackResult] = []

        # Map tracked bbox to face detections
        for tid, raw_bbox, state in tracked:
            state.face_eval_count += 1
            smoothed_box = state.smooth_bbox(raw_bbox, timestamp)

            # Find matching face detection by IoU / center distance
            matching_face = None
            for f in faces:
                if f.landmarks is not None:
                    fcx = (f.bbox[0] + f.bbox[2]) / 2.0
                    fcy = (f.bbox[1] + f.bbox[3]) / 2.0
                    if raw_bbox[0] <= fcx <= raw_bbox[2] and raw_bbox[1] <= fcy <= raw_bbox[3]:
                        matching_face = f
                        break

            if matching_face is not None and matching_face.landmarks is not None:
                state.consecutive_landmark_misses = 0
                face_box = state.smooth_face_bbox(matching_face.bbox, timestamp)
                landmarks = matching_face.landmarks
                state.last_face_box = face_box
                state.last_landmarks = landmarks

                passed, reason, metrics = filter_5_landmarks(
                    landmarks=landmarks,
                    face_bbox=face_box,
                    face_conf=matching_face.conf if matching_face is not None else 1.0,
                    cfg=self.cfg.landmark_filter,
                )

                if passed:
                    # Align once to 112x112
                    aligned, _ = self.recognizer.align(frame, landmarks)
                    if hasattr(self.quality_estimator, "score_aligned"):
                        q_score = self.quality_estimator.score_aligned(aligned)
                    else:
                        q_score = self.quality_estimator.score(frame, landmarks)
                    is_learning_target = (tid == self.locked_track_id and not state.gallery.is_complete)
                    should_recognize = False
                    if q_score >= self.cfg.quality.min_quality:
                        if state.person_id is None or is_learning_target:
                            should_recognize = True
                        elif q_score > (state.best_quality + self.cfg.quality.upgrade_delta):
                            should_recognize = True
                        elif (timestamp - state.last_recognition_time) >= self.cfg.recognition.recheck_interval_s:
                            should_recognize = True

                    if should_recognize:
                        state.best_quality = max(state.best_quality, q_score)
                        state.last_recognition_time = timestamp
                        emb = self.recognizer.extract_embedding(aligned)
                        state.best_embedding = emb
                        state.best_landmarks = landmarks
                        state.best_face_crop = aligned

                        match_meta, sim = self.recognizer.search(emb, threshold=self.cfg.recognition.match_thr)
                        if match_meta is not None:
                            pid = match_meta.get("person_id")
                            name = match_meta.get("full_name", match_meta.get("name", "Known"))
                            state.vote(pid=pid, name=name, sim=sim)
                        else:
                            state.vote(pid=None, name="Unknown", sim=sim)

                        target_pid = state.person_id or (self.locked_person_id if self.locked_track_id == tid else None)
                        if target_pid is not None and self.recognizer.is_person_valid(target_pid):
                            if state.gallery.anchor_embedding is None:
                                state.gallery.set_anchor(emb, q_score, aligned)
                            if not state.gallery.is_complete:
                                eval_res = self.recognizer.evaluate_distinctiveness(emb, target_person_id=target_pid)
                                committed = state.gallery.propose_candidate(emb, q_score, aligned, eval_res, timestamp)
                                if committed is not None:
                                    norm_emb, q, crop = committed
                                    self.recognizer.add_template(norm_emb, {"person_id": target_pid, "name": state.person_name})
                                    if self.on_template_learned:
                                        self.on_template_learned(target_pid, norm_emb, q, crop)
            else:
                state.consecutive_landmark_misses += 1
                face_box = state.smooth_face_bbox(smoothed_box, timestamp)
                if state.last_landmarks is not None and state.last_face_box is not None and state.consecutive_landmark_misses <= 10:
                    landmarks = propagate_landmarks_motion(state.last_landmarks, state.last_face_box, face_box)
                    state.last_face_box = face_box
                    state.last_landmarks = landmarks
                else:
                    landmarks = None

            res = TrackResult(
                track_id=tid,
                raw_bbox=raw_bbox,
                smoothed_bbox=smoothed_box,
                person_id=state.person_id,
                person_name=state.person_name,
                similarity=state.similarity,
                quality=state.best_quality,
                is_unknown=state.is_unknown,
                face_bbox=face_box,
                landmarks=landmarks,
                lost_frames=state.lost_frames,
                timestamp=timestamp,
                is_locked=False,
                learning_count=state.gallery.total_count,
                learning_target=state.gallery.target_quota,
                learning_margin=state.gallery.last_margin,
                is_learning_complete=state.gallery.is_complete,
            )
            results.append(res)


        return self._enforce_exclusive_identities_and_locking(results)


def render_overlay(
    frame: np.ndarray,
    results: List[TrackResult],
    cfg: OverlayConfig,
    fps: float = 0.0,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """Renders professional bounding boxes, landmarks, identity tags, and status HUD."""
    if frame is None or frame.size == 0:
        return frame

    out = frame.copy()

    for res in results:
        # Determine person box color: Gold for locked, Green for known, Blue/Orange for unknown
        if res.is_locked:
            color = (0, 215, 255)   # Bright Gold / Amber for locked target
            box_thick = 3
        elif res.is_unknown:
            color = (30, 144, 255)  # Orange/Blue in BGR
            box_thick = 2
        else:
            color = (50, 205, 50)   # Lime Green
            box_thick = 2

        # Draw Person / Track Bounding Box
        if cfg.show_person_bbox:
            box = res.smoothed_bbox
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick, cv2.LINE_AA)

            # Draw tactical corner reticles for locked target
            if res.is_locked:
                c_len = min(24, max(8, int(min(x2 - x1, y2 - y1) * 0.15)))
                # Top-Left
                cv2.line(out, (x1 - 2, y1 - 2), (x1 + c_len, y1 - 2), color, 4, cv2.LINE_AA)
                cv2.line(out, (x1 - 2, y1 - 2), (x1 - 2, y1 + c_len), color, 4, cv2.LINE_AA)
                # Top-Right
                cv2.line(out, (x2 + 2, y1 - 2), (x2 - c_len, y1 - 2), color, 4, cv2.LINE_AA)
                cv2.line(out, (x2 + 2, y1 - 2), (x2 + 2, y1 + c_len), color, 4, cv2.LINE_AA)
                # Bottom-Left
                cv2.line(out, (x1 - 2, y2 + 2), (x1 + c_len, y2 + 2), color, 4, cv2.LINE_AA)
                cv2.line(out, (x1 - 2, y2 + 2), (x1 - 2, y2 - c_len), color, 4, cv2.LINE_AA)
                # Bottom-Right
                cv2.line(out, (x2 + 2, y2 + 2), (x2 - c_len, y2 + 2), color, 4, cv2.LINE_AA)
                cv2.line(out, (x2 + 2, y2 + 2), (x2 + 2, y2 - c_len), color, 4, cv2.LINE_AA)
                # Center reticle crosshair
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

            # Draw Label Tag Header
            label_parts = []
            if res.is_locked:
                label_parts.append("[LOCKED]")
            if cfg.show_id:
                label_parts.append(f"#{res.track_id}")
            if cfg.show_name:
                label_parts.append(res.person_name)
            if cfg.show_quality and res.quality > 0:
                label_parts.append(f"Q:{res.quality:.2f}")
            if not res.is_unknown and res.similarity > 0:
                label_parts.append(f"{int(res.similarity * 100)}%")

            label_text = " ".join(label_parts)
            if label_text:
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                tag_y1 = max(0, y1 - th - 8)
                tag_y2 = y1
                tag_x2 = min(out.shape[1], x1 + tw + 10)
                cv2.rectangle(out, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
                tag_text_color = (0, 0, 0) if res.is_locked else (255, 255, 255)
                cv2.putText(
                    out,
                    label_text,
                    (x1 + 5, tag_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    tag_text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Draw tactical learning badge below target box for locked target
            if res.is_locked:
                if res.is_learning_complete:
                    sub_text = f"🎯 TARGET READY ({res.learning_count}/{res.learning_target})"
                    sub_bg = (46, 160, 67)  # Forest Green
                    sub_fg = (255, 255, 255)
                else:
                    margin_str = f"+{res.learning_margin:.2f}" if res.learning_margin >= 0 else f"{res.learning_margin:.2f}"
                    sub_text = f"🎯 ADAPTING {res.learning_count}/{res.learning_target} (Margin: {margin_str})"
                    sub_bg = (0, 180, 230)  # Amber / Gold
                    sub_fg = (0, 0, 0)

                (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                sub_y1 = min(out.shape[0] - sth - 8, y2 + 3)
                sub_y2 = sub_y1 + sth + 6
                sub_x2 = min(out.shape[1], x1 + stw + 10)
                cv2.rectangle(out, (x1, sub_y1), (sub_x2, sub_y2), sub_bg, -1)
                cv2.putText(
                    out,
                    sub_text,
                    (x1 + 5, sub_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    sub_fg,
                    1,
                    cv2.LINE_AA,
                )

        # Draw Face Bounding Box
        if cfg.show_face_bbox and res.face_bbox is not None:
            fx1, fy1, fx2, fy2 = (
                int(res.face_bbox[0]),
                int(res.face_bbox[1]),
                int(res.face_bbox[2]),
                int(res.face_bbox[3]),
            )
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 215, 0), 1, cv2.LINE_AA)

        # Draw 5 Facial Landmarks
        if cfg.show_landmarks and res.landmarks is not None:
            pts = np.asarray(res.landmarks, dtype=int)
            landmark_colors = [
                (0, 0, 255),   # Left Eye: Red
                (0, 0, 255),   # Right Eye: Red
                (0, 255, 255), # Nose: Yellow
                (255, 0, 0),   # Left Mouth: Blue
                (255, 0, 0),   # Right Mouth: Blue
            ]
            for i, (lx, ly) in enumerate(pts):
                pt_color = landmark_colors[i] if i < len(landmark_colors) else (0, 255, 0)
                cv2.circle(out, (int(lx), int(ly)), 3, pt_color, -1, cv2.LINE_AA)

    # Draw HUD (FPS, Latency, Track count)
    if cfg.show_fps or cfg.show_latency:
        hud_parts = []
        if cfg.show_fps:
            hud_parts.append(f"FPS: {fps:.1f}")
        if cfg.show_latency:
            hud_parts.append(f"{latency_ms:.1f} ms")
        hud_parts.append(f"Tracks: {len(results)}")
        hud_str = " | ".join(hud_parts)

        (hw, hh), hbase = cv2.getTextSize(hud_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (10, 10), (10 + hw + 12, 10 + hh + 10), (20, 20, 20), -1)
        cv2.rectangle(out, (10, 10), (10 + hw + 12, 10 + hh + 10), (60, 60, 60), 1)
        cv2.putText(out, hud_str, (16, 10 + hh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return out
