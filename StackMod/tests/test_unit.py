"""Unit tests UT-01 to UT-10 for individual modules and core algorithms."""
from __future__ import annotations

import math
import os
import shutil
import numpy as np
import pytest

from core.config import (
    AppConfig,
    LandmarkFilterConfig,
    RecognitionConfig,
    SmoothingConfig,
)
from core.smoothing import OneEuro, xyxy_to_cxcywh, cxcywh_to_xyxy, BoxSmoother
from core.quality import calculate_landmark_geometry, filter_5_landmarks
from core.detectors import FaceDetectionResult, offset_to_frame
from core.tracker import TrackState
from core.recognizer import FaceRecognizer
from db.repo import DatabaseRepo
from uniface import Face


def test_ut01_landmark_frontal_ok():
    """UT-01: Ideal frontal 5 landmarks pass filter."""
    landmarks = np.array([
        [100.0, 100.0],  # Left Eye
        [150.0, 100.0],  # Right Eye
        [125.0, 125.0],  # Nose (centered)
        [110.0, 150.0],  # Left Mouth
        [140.0, 150.0],  # Right Mouth
    ], dtype=np.float32)
    bbox = np.array([80.0, 70.0, 170.0, 170.0], dtype=np.float32)
    cfg = LandmarkFilterConfig(enabled=True, min_face_size=50.0, max_yaw_ratio=0.35, max_roll_deg=25.0)

    passed, reason, metrics = filter_5_landmarks(landmarks, bbox, face_conf=0.9, cfg=cfg)
    assert passed, f"Frontal landmarks failed filter: {reason}"
    assert metrics["yaw_ratio"] < 0.1
    assert metrics["roll_deg"] < 1.0


def test_ut02_landmark_yaw_rejection():
    """UT-02: Extreme yaw angle fails filter."""
    # Nose severely shifted to the far right
    landmarks = np.array([
        [100.0, 100.0],  # Left Eye
        [150.0, 100.0],  # Right Eye
        [148.0, 125.0],  # Nose (extreme yaw)
        [120.0, 150.0],
        [150.0, 150.0],
    ], dtype=np.float32)
    bbox = np.array([80.0, 70.0, 170.0, 170.0], dtype=np.float32)
    cfg = LandmarkFilterConfig(enabled=True, min_face_size=50.0, max_yaw_ratio=0.30)

    passed, reason, metrics = filter_5_landmarks(landmarks, bbox, face_conf=0.9, cfg=cfg)
    assert not passed
    assert "Yaw ratio" in reason


def test_ut03_landmark_small_face_rejection():
    """UT-03: Face below minimum size fails filter."""
    landmarks = np.array([
        [10.0, 10.0],
        [20.0, 10.0],
        [15.0, 15.0],
        [12.0, 20.0],
        [18.0, 20.0],
    ], dtype=np.float32)
    bbox = np.array([5.0, 5.0, 25.0, 25.0], dtype=np.float32)  # width = 20px
    cfg = LandmarkFilterConfig(enabled=True, min_face_size=60.0)

    passed, reason, _ = filter_5_landmarks(landmarks, bbox, face_conf=0.9, cfg=cfg)
    assert not passed
    assert "Face size" in reason


def test_ut04_one_euro_filter_jitter_reduction():
    """UT-04: One Euro filter reduces standard deviation on noisy stationary signal."""
    filt = OneEuro(min_cutoff=0.5, beta=0.005, d_cutoff=1.0)
    np.random.seed(42)
    true_val = np.array([100.0, 200.0])
    noise = np.random.normal(0, 2.0, size=(100, 2))
    raw_signals = true_val + noise

    filtered = []
    for i, s in enumerate(raw_signals):
        filtered.append(filt(s, t=i * 0.033))

    filtered = np.array(filtered)
    raw_std = np.std(raw_signals[20:], axis=0)
    filt_std = np.std(filtered[20:], axis=0)

    assert np.all(filt_std < raw_std), f"Filtered std {filt_std} should be lower than raw std {raw_std}"


def test_ut05_one_euro_step_response():
    """UT-05: One Euro filter adapts rapidly to sudden step changes."""
    filt = OneEuro(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    # 20 frames at 100, then jump to 500
    times = [i * 0.033 for i in range(40)]
    vals = [100.0] * 20 + [500.0] * 20
    out = []
    for t, v in zip(times, vals):
        out.append(float(filt(np.array([v]), t)[0]))

    # Step happened at frame 20, by frame 26 it should be close to 500
    assert out[26] > 450.0, f"Filter lagged too much on step change: {out[26]}"


def test_ut06_track_state_voting_confirmation():
    """UT-06: Voting consensus confirms identity when matching vote_min."""
    recog_cfg = RecognitionConfig(vote_window=7, vote_min=3)
    smooth_cfg = SmoothingConfig()
    st = TrackState(track_id=1, recog_cfg=recog_cfg, smooth_cfg=smooth_cfg)

    st.vote(pid=10, name="Alice", sim=0.85)
    assert st.person_id is None  # 1 vote, not enough
    st.vote(pid=10, name="Alice", sim=0.88)
    assert st.person_id is None  # 2 votes, not enough
    st.vote(pid=10, name="Alice", sim=0.90)
    assert st.person_id == 10  # 3 votes -> Confirmed!
    assert st.person_name == "Alice"
    assert not st.is_unknown


def test_ut07_track_state_voting_inconsistent():
    """UT-07: Inconsistent alternating votes do not confirm false identity."""
    recog_cfg = RecognitionConfig(vote_window=7, vote_min=3)
    smooth_cfg = SmoothingConfig()
    st = TrackState(track_id=2, recog_cfg=recog_cfg, smooth_cfg=smooth_cfg)

    st.vote(pid=1, name="Alice", sim=0.7)
    st.vote(pid=2, name="Bob", sim=0.7)
    st.vote(pid=1, name="Alice", sim=0.7)
    st.vote(pid=2, name="Bob", sim=0.7)

    assert st.person_id is None  # No identity has >= 3 votes


def test_ut08_offset_to_frame():
    """UT-08: Crop coordinates shift properly to frame coordinates."""
    face = Face(
        bbox=np.array([10.0, 15.0, 60.0, 65.0]),
        landmarks=np.array([[20.0, 25.0], [40.0, 25.0], [30.0, 35.0], [25.0, 50.0], [35.0, 50.0]]),
        confidence=0.95,
    )
    res = offset_to_frame(face, offset_x=100, offset_y=200)

    assert np.allclose(res.bbox, [110.0, 215.0, 160.0, 265.0])
    assert np.allclose(res.landmarks[0], [120.0, 225.0])
    assert res.conf == 0.95


@pytest.mark.slow
def test_ut09_embedding_normalization(tmp_path):
    """UT-09: Extracted embedding has unit L2 norm."""
    cfg = RecognitionConfig()
    recognizer = FaceRecognizer(cfg, faiss_path=str(tmp_path / "test_unit_faiss"))
    # 112x112 synthetic face crop
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb = recognizer.extract_embedding(dummy_face)

    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm {norm} is not 1.0"
    assert emb.shape == (512,)


def test_ut10_repo_enroll_and_faiss_sync(tmp_path):
    """UT-10: Enroll person in DB and verify FAISS search match."""
    db_path = f"sqlite:///{tmp_path}/test_ut10.db"
    faiss_dir = str(tmp_path / "test_ut10_faiss")
    snapshots_dir = str(tmp_path / "test_snapshots")

    repo = DatabaseRepo(db_path)
    recog = FaceRecognizer(RecognitionConfig(), faiss_path=faiss_dir)

    # Generate synthetic 512-dim embedding
    np.random.seed(123)
    dummy_emb = np.random.randn(512).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)
    dummy_crop = np.full((112, 112, 3), 150, dtype=np.uint8)

    person = repo.enroll_face(
        code="NV999",
        full_name="Nguyễn Văn Test",
        group_name="QA",
        templates=[(dummy_emb, 0.85, dummy_crop)],
        recognizer=recog,
        snapshots_dir=snapshots_dir,
    )

    assert person.code == "NV999"
    assert person.full_name == "Nguyễn Văn Test"

    # Search with same embedding
    meta, sim = recog.search(dummy_emb, threshold=0.5)
    assert meta is not None
    assert meta["person_id"] == person.id
    assert meta["code"] == "NV999"
    assert sim > 0.99

    # Test sync_faiss
    count = repo.sync_faiss(recog)
    assert count == 1


def test_ut11_track_lock_state():
    """UT-11: Verify Pipeline target locking and unlocking mechanism."""
    from core.pipeline import Pipeline
    from core.config import AppConfig

    pipe = Pipeline(AppConfig())
    assert pipe.locked_track_id is None
    assert pipe.locked_person_id is None

    pipe.lock_target(track_id=42, person_id=1)
    assert pipe.locked_track_id == 42
    assert pipe.locked_person_id == 1

    pipe.unlock_target()
    assert pipe.locked_track_id is None
    assert pipe.locked_person_id is None


def test_ut12_locked_render_overlay():
    """UT-12: Verify render_overlay draws tactical target reticle for locked tracks."""
    from core.pipeline import render_overlay
    from core.tracker import TrackResult
    from core.config import OverlayConfig
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    locked_res = TrackResult(
        track_id=1,
        raw_bbox=np.array([100, 100, 300, 400], dtype=np.float32),
        smoothed_bbox=np.array([100, 100, 300, 400], dtype=np.float32),
        person_id=1,
        person_name="Person 1",
        similarity=0.95,
        quality=0.88,
        is_unknown=False,
        is_locked=True,
    )
    unlocked_res = TrackResult(
        track_id=2,
        raw_bbox=np.array([500, 100, 700, 400], dtype=np.float32),
        smoothed_bbox=np.array([500, 100, 700, 400], dtype=np.float32),
        person_id=None,
        person_name="Unknown",
        similarity=0.0,
        quality=0.5,
        is_unknown=True,
        is_locked=False,
    )

    out = render_overlay(frame, [locked_res, unlocked_res], OverlayConfig())
    assert out.shape == frame.shape
    # Check that drawing actually modified pixels inside the canvas
    assert np.any(out > 0)


def test_ut13_face_gallery_learning_and_staging():
    """UT-13: Verify FaceGallery 5-sample quota, 2-stage staging buffer, and completion."""
    from core.tracker import FaceGallery
    import numpy as np

    gallery = FaceGallery(target_quota=5)
    assert gallery.total_count == 0
    assert not gallery.is_complete

    # 1. Set Anchor
    v_anchor = np.zeros(512, dtype=np.float32)
    v_anchor[0] = 1.0
    gallery.set_anchor(v_anchor, quality=0.90)
    assert gallery.total_count == 1
    assert not gallery.is_complete

    # 2. Propose candidate 1: Frame 1 buffers into staging
    v_cand1 = 0.85 * v_anchor.copy()
    v_cand1[1] = np.sqrt(1 - 0.85**2)
    eval_ok = {"is_distinct": True, "margin": 0.35}

    res1 = gallery.propose_candidate(v_cand1, quality=0.85, crop=None, eval_result=eval_ok, now=1.0)
    assert res1 is None  # Held in staging
    assert gallery.total_count == 1

    # 3. Propose candidate 1: Frame 2 confirms with high pair_sim
    res2 = gallery.propose_candidate(v_cand1, quality=0.86, crop=None, eval_result=eval_ok, now=1.1)
    assert res2 is not None  # Committed!
    assert gallery.total_count == 2

    # 4. Fill remaining 3 diverse samples
    for i in range(2, 5):
        v = 0.80 * v_anchor.copy()
        v[i] = np.sqrt(1 - 0.80**2)
        # Frame 1
        gallery.propose_candidate(v, quality=0.85, crop=None, eval_result=eval_ok, now=2.0 + i)
        # Frame 2 confirm
        committed = gallery.propose_candidate(v, quality=0.87, crop=None, eval_result=eval_ok, now=2.1 + i)
        assert committed is not None

    assert gallery.total_count == 5
    assert gallery.is_complete

    # 5. Over-quota candidate is rejected once complete
    extra_cand = v_anchor.copy()
    rejected = gallery.propose_candidate(extra_cand, quality=0.95, crop=None, eval_result=eval_ok, now=10.0)
    assert rejected is None
    assert gallery.total_count == 5


def test_ut14_evaluate_distinctiveness_margin(tmp_path):
    """UT-14: Verify distinctiveness margin evaluation against existing IDs in FAISS."""
    from core.recognizer import FaceRecognizer
    from core.config import RecognitionConfig
    import numpy as np

    faiss_dir = str(tmp_path / "test_ut14_faiss")
    recog = FaceRecognizer(RecognitionConfig(), faiss_path=faiss_dir)

    # Enroll Person 1
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    recog.add_template(v1, {"person_id": 1, "name": "Person 1"})

    # Enroll Person 2 (Orthogonal)
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    recog.add_template(v2, {"person_id": 2, "name": "Person 2"})

    # Candidate 1: Close to Person 1, far from Person 2 (distinct, high margin)
    query_p1 = 0.88 * v1 + np.sqrt(1 - 0.88**2) * np.roll(v1, 2)
    eval_p1 = recog.evaluate_distinctiveness(query_p1, target_person_id=1, margin_thr=0.15)
    assert eval_p1["is_distinct"] is True
    assert eval_p1["sim_self"] > 0.85
    assert eval_p1["margin"] > 0.50

    # Candidate 2: Dangerously close to Person 2 while belonging to Person 1 (ambiguous, rejected)
    query_ambiguous = 0.65 * v1 + 0.65 * v2
    query_ambiguous /= np.linalg.norm(query_ambiguous)
    eval_amb = recog.evaluate_distinctiveness(query_ambiguous, target_person_id=1, margin_thr=0.15)
    assert eval_amb["is_distinct"] is False
    assert eval_amb["margin"] < 0.15


def test_ut15_face_bbox_smoothing():
    """UT-15: Verify TrackState.smooth_face_bbox dampens high-frequency jitter."""
    recog_cfg = RecognitionConfig()
    smooth_cfg = SmoothingConfig(enabled=True, min_cutoff=1.0, beta=0.01)
    state = TrackState(track_id=1, recog_cfg=recog_cfg, smooth_cfg=smooth_cfg)

    # Base face box: [100, 100, 180, 180]
    base_box = np.array([100.0, 100.0, 180.0, 180.0], dtype=np.float32)
    s1 = state.smooth_face_bbox(base_box, timestamp=1.0)
    assert np.allclose(s1, base_box)

    # Add rapid artificial noise +/- 10px jitter
    noisy_box = np.array([110.0, 95.0, 190.0, 175.0], dtype=np.float32)
    s2 = state.smooth_face_bbox(noisy_box, timestamp=1.033)
    # Filtered output should be dampened between base and noisy box
    assert s2[0] < noisy_box[0]
    assert s2[0] > base_box[0]
    # Variance/jitter is reduced
    jitter_raw = abs(noisy_box[0] - base_box[0])
    jitter_smoothed = abs(s2[0] - base_box[0])
    assert jitter_smoothed < jitter_raw


def test_ut16_propagate_landmarks_motion():
    """UT-16: Verify propagate_landmarks_motion translates and scales 5 landmarks with box displacement."""
    from core.pipeline import propagate_landmarks_motion

    old_box = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
    old_landmarks = np.array([
        [125.0, 125.0],  # Left Eye
        [175.0, 125.0],  # Right Eye
        [150.0, 150.0],  # Nose (center)
        [135.0, 175.0],  # Left Mouth
        [165.0, 175.0],  # Right Mouth
    ], dtype=np.float32)

    # Case 1: Pure translation (+20px X, +10px Y)
    new_box_trans = np.array([120.0, 110.0, 220.0, 210.0], dtype=np.float32)
    new_lm_trans = propagate_landmarks_motion(old_landmarks, old_box, new_box_trans)
    assert np.allclose(new_lm_trans[2], [170.0, 160.0])  # Nose shifted exactly (+20, +10)
    assert np.allclose(new_lm_trans[:, 0] - old_landmarks[:, 0], 20.0)
    assert np.allclose(new_lm_trans[:, 1] - old_landmarks[:, 1], 10.0)

    # Case 2: Scale expansion (2x size around center)
    new_box_scale = np.array([50.0, 50.0, 250.0, 250.0], dtype=np.float32)  # size 200x200 (2x)
    new_lm_scale = propagate_landmarks_motion(old_landmarks, old_box, new_box_scale)
    # Center (nose) remains at [150, 150]
    assert np.allclose(new_lm_scale[2], [150.0, 150.0])
    # Eye distance doubles from 50px to 100px
    old_eye_dist = old_landmarks[1, 0] - old_landmarks[0, 0]
    new_eye_dist = new_lm_scale[1, 0] - new_lm_scale[0, 0]
    assert np.isclose(new_eye_dist, old_eye_dist * 2.0)


def test_ut17_target_lock_spatial_continuity():
    """UT-17: Verify target lock seamlessly migrates across ByteTrack track ID switches via spatial IoU hand-off."""
    from core.pipeline import Pipeline
    from core.tracker import TrackResult

    pipe = Pipeline(AppConfig())
    # User locks onto Track 1
    pipe.lock_target(track_id=1)
    assert pipe.locked_track_id == 1

    # Frame 1: Track 1 is active at [100, 100, 300, 400]
    res1 = TrackResult(
        track_id=1,
        raw_bbox=np.array([100, 100, 300, 400], dtype=np.float32),
        smoothed_bbox=np.array([100, 100, 300, 400], dtype=np.float32),
        person_id=None,
        person_name="Unknown",
        similarity=0.0,
        is_unknown=True,
    )
    out1 = pipe._enforce_exclusive_identities_and_locking([res1])
    assert out1[0].is_locked is True
    assert pipe.locked_track_id == 1

    # Frame 2: ByteTrack switches ID: Track 1 disappears, Track 2 appears at almost same box [105, 102, 305, 402]
    res2 = TrackResult(
        track_id=2,
        raw_bbox=np.array([105, 102, 305, 402], dtype=np.float32),
        smoothed_bbox=np.array([105, 102, 305, 402], dtype=np.float32),
        person_id=None,
        person_name="Unknown",
        similarity=0.0,
        is_unknown=True,
    )
    out2 = pipe._enforce_exclusive_identities_and_locking([res2])
    # Spatial continuity should migrate lock to Track 2!
    assert out2[0].is_locked is True
    assert pipe.locked_track_id == 2


def test_ut18_normalize_stream_uri():
    """UT-18: Verify normalize_stream_uri formats DroidCam & RTSP URIs correctly."""
    from core.sources import normalize_stream_uri

    assert normalize_stream_uri("http://192.168.1.50:4747") == "http://192.168.1.50:4747/video"
    assert normalize_stream_uri("http://192.168.1.50:4747/") == "http://192.168.1.50:4747/video"
    assert normalize_stream_uri("192.168.1.50:4747") == "http://192.168.1.50:4747/video"
    assert normalize_stream_uri("http://192.168.1.50:4747/video") == "http://192.168.1.50:4747/video"
    assert normalize_stream_uri("http://192.168.1.50:4747/mjpegfeed") == "http://192.168.1.50:4747/mjpegfeed"
    assert normalize_stream_uri("rtsp://admin:pass@192.168.1.20:554/stream") == "rtsp://admin:pass@192.168.1.20:554/stream"


def test_ut19_stream_source_open_no_unsupported_params(monkeypatch):
    """UT-19: Verify StreamSource does not pass CAP_PROP_BUFFERSIZE in open params (which triggers FFMPEG bailout)."""
    import cv2
    from core.sources import StreamSource

    captured_params = []

    class MockCap:
        def __init__(self, uri, api=None, params=None):
            if params is not None:
                captured_params.append(list(params))
        def isOpened(self):
            return True
        def set(self, prop, val):
            return True
        def read(self):
            return True, np.zeros((100, 100, 3), dtype=np.uint8)
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", MockCap)

    src = StreamSource("http://192.168.1.50:4747")
    try:
        # Give worker a split second to run
        import time
        time.sleep(0.05)
        assert len(captured_params) > 0
        params = captured_params[0]
        # CAP_PROP_BUFFERSIZE must NOT be in open params!
        assert cv2.CAP_PROP_BUFFERSIZE not in params
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params
        assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in params
    finally:
        src.stop()


def test_ut20_locked_target_immediate_anchor_and_continuous_learning(tmp_path):
    """UT-20: Verify target lock immediate anchor (1/5 not 0/5), max_self_sim 0.96, and 0.3s cooldown."""
    from core.tracker import FaceGallery
    from core.recognizer import FaceRecognizer
    from core.config import RecognitionConfig
    import numpy as np

    # 1. Immediate anchor initialization: count starts at 1/5
    gallery = FaceGallery(target_quota=5)
    assert gallery.total_count == 0  # Uninitialized
    v_anchor = np.zeros(512, dtype=np.float32)
    v_anchor[0] = 1.0
    gallery.set_anchor(v_anchor, quality=0.88)
    assert gallery.total_count == 1  # Immediately 1/5 upon lock!
    assert not gallery.is_complete

    # 2. Distinctiveness evaluation with max_self_sim = 0.96:
    # A subject facing camera in real time has sim_self ~ 0.94.
    # With max_self_sim = 0.96, it is recognized as distinct (not rejected as redundant).
    faiss_dir = str(tmp_path / "test_ut20_faiss")
    recog = FaceRecognizer(RecognitionConfig(), faiss_path=faiss_dir)
    recog.add_template(v_anchor, {"person_id": 1, "name": "Person 1"})

    # Query with natural in-motion variation (sim ~ 0.94)
    v_query = 0.94 * v_anchor.copy()
    v_query[2] = np.sqrt(1 - 0.94**2)
    eval_res = recog.evaluate_distinctiveness(v_query, target_person_id=1)
    assert eval_res["is_distinct"] is True
    assert eval_res["is_redundant"] is False
    assert eval_res["sim_self"] > 0.93

    # 3. Two-stage proposal and 0.3s cooldown
    eval_ok = {"is_distinct": True, "margin": 0.40}
    # Frame 1: staging buffer
    c1 = gallery.propose_candidate(v_query, quality=0.85, crop=None, eval_result=eval_ok, now=1.0)
    assert c1 is None
    # Frame 2: consecutive confirm -> commit
    c2 = gallery.propose_candidate(v_query, quality=0.87, crop=None, eval_result=eval_ok, now=1.05)
    assert c2 is not None
    assert gallery.total_count == 2
    assert gallery.learn_block_until >= 1.35  # 1.05 + 0.3s cooldown

    # Frame 3 within cooldown (< 1.35) is rejected
    c3 = gallery.propose_candidate(v_query, quality=0.90, crop=None, eval_result=eval_ok, now=1.20)
    assert c3 is None
    assert gallery.total_count == 2


def test_ut21_video_widget_side_tags_and_face_avatar_thumbnails(qapp):
    """UT-21: Verify VideoWidget extracts face thumbnails and renders side tags without error."""
    from ui.video_widget import VideoWidget
    from core.tracker import TrackResult
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtCore import Qt

    widget = VideoWidget()
    widget.resize(800, 450)

    # Create synthetic frame & raw_frame
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    raw_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

    # TrackResult with locked target and face_bbox
    res_locked = TrackResult(
        track_id=1,
        raw_bbox=np.array([100.0, 100.0, 300.0, 400.0]),
        smoothed_bbox=np.array([100.0, 100.0, 300.0, 400.0]),
        person_id=1,
        person_name="Person 1",
        similarity=0.92,
        quality=0.88,
        is_unknown=False,
        face_bbox=np.array([150.0, 120.0, 250.0, 220.0]),
        is_locked=True,
        learning_count=3,
        learning_target=5,
        is_learning_complete=False,
    )
    res_other = TrackResult(
        track_id=2,
        raw_bbox=np.array([350.0, 100.0, 500.0, 400.0]),
        smoothed_bbox=np.array([350.0, 100.0, 500.0, 400.0]),
        person_id=None,
        person_name="Unknown",
        similarity=0.0,
        quality=0.75,
        is_unknown=True,
        face_bbox=np.array([380.0, 120.0, 460.0, 200.0]),
        is_locked=False,
    )

    widget.update_frame(frame, results=[res_locked, res_other], raw_frame=raw_frame)
    assert 1 in widget._face_thumbnails
    assert 2 in widget._face_thumbnails

    # Trigger repaint on wide canvas (dest_x >= 110)
    canvas = QPixmap(1000, 500)
    canvas.fill(Qt.black)
    p = QPainter(canvas)
    widget._draw_tracking_tags(p, 1000, 500, 180, 10, 640, 480)
    p.end()

    # Trigger repaint on narrow canvas (floating HUD fallback)
    p2 = QPainter(canvas)
    widget._draw_tracking_tags(p2, 640, 480, 0, 0, 640, 480)
    p2.end()


def test_ut22_tuning_dock_apply_config_and_persistence(qapp, tmp_path):
    """UT-22: Verify TuningDock loads AppConfig values and persists to disk."""
    from ui.docks.tuning_dock import TuningDock
    from core.config import AppConfig

    custom_cfg = AppConfig()
    custom_cfg.recognition.match_thr = 0.75
    custom_cfg.quality.min_quality = 0.55
    custom_cfg.smoothing.min_cutoff = 2.5
    custom_cfg.smoothing.beta = 0.045
    custom_cfg.landmark_filter.max_yaw_ratio = 0.30

    dock = TuningDock(cfg=custom_cfg)
    assert dock.match_slider.value() == 75
    assert dock.quality_slider.value() == 55
    assert dock.cutoff_slider.value() == 25
    assert dock.beta_slider.value() == 45
    assert dock.yaw_slider.value() == 30
    assert dock.match_val_label.text() == "0.75"

    # Save to disk and reload
    cfg_file = str(tmp_path / "saved_config.json")
    custom_cfg.save_json(cfg_file)
    loaded_cfg = AppConfig.load_json(cfg_file)
    assert loaded_cfg.recognition.match_thr == 0.75
    assert loaded_cfg.smoothing.beta == 0.045


def test_ut23_reid_adaptation_and_candidate_staging(tmp_path):
    """UT-23: Verify FaceGallery staging persistence on redundancy, pair_sim commit, and non-ambiguous multi-ID evaluation."""
    from core.tracker import FaceGallery
    from core.recognizer import FaceRecognizer
    from core.config import AppConfig

    # 1. Config defaults verification
    cfg = AppConfig()
    assert cfg.recognition.match_thr == 0.50
    assert cfg.recognition.vote_min == 2
    assert cfg.recognition.hysteresis_thr == 0.42

    # 2. FaceGallery staging persistence
    gallery = FaceGallery(target_quota=5)
    v_anchor = np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32)
    gallery.set_anchor(v_anchor, quality=0.85)
    assert gallery.total_count == 1

    # Stage candidate frame 1 (distinct angle, e.g. 15-deg yaw)
    v_cand1 = np.array([0.92, 0.39, 0.0] + [0.0] * 509, dtype=np.float32)
    v_cand1 /= np.linalg.norm(v_cand1)
    eval_distinct = {"is_distinct": True, "margin": 0.30}
    res1 = gallery.propose_candidate(v_cand1, 0.82, None, eval_distinct, now=10.0)
    assert res1 is None
    assert gallery.staging_emb is not None

    # Frame 2 is redundant with DB (e.g. facing back towards anchor), but NOT mismatch or ambiguous
    eval_redundant = {"is_distinct": False, "is_redundant": True, "is_mismatch": False, "is_ambiguous": False}
    res2 = gallery.propose_candidate(v_anchor, 0.88, None, eval_redundant, now=10.05)
    assert res2 is None
    # Verify staging_emb is preserved and NOT wiped out
    assert gallery.staging_emb is not None

    # Frame 3 arrives at same candidate angle (pair_sim >= 0.75) and is distinct
    v_cand2 = np.array([0.91, 0.41, 0.0] + [0.0] * 509, dtype=np.float32)
    v_cand2 /= np.linalg.norm(v_cand2)
    committed = gallery.propose_candidate(v_cand2, 0.84, None, eval_distinct, now=10.10)
    assert committed is not None
    assert gallery.total_count == 2
    assert gallery.staging_emb is None

    # 3. Recognizer evaluate_distinctiveness with duplicate IDs
    from core.config import RecognitionConfig
    recog = FaceRecognizer(cfg=RecognitionConfig(), faiss_path=str(tmp_path / "faiss.index"))
    # Add template for Person 1 and Person 2 (both representing duplicate enrollments of user)
    recog.add_template(v_anchor, {"person_id": 1, "name": "Person 1"})
    v_p2 = np.array([0.98, 0.19, 0.0] + [0.0] * 509, dtype=np.float32)
    v_p2 /= np.linalg.norm(v_p2)
    recog.add_template(v_p2, {"person_id": 2, "name": "Person 2"})

    # Evaluate target_person_id = 2 with incoming frame that is close to both Person 1 and 2
    eval_dup = recog.evaluate_distinctiveness(v_cand1, target_person_id=2)
    # Even though Person 1 and Person 2 are close in DB, it is not flagged as ambiguous
    assert eval_dup["is_ambiguous"] is False


def test_ut24_nvidia_gpu_hardware_detection():
    """UT-24: Verify NVIDIA GPU detection, session options, and provider fallbacks."""
    from core.config import (
        get_gpu_device_name,
        get_cuda_session_options,
        get_dml_session_options,
        get_optimal_providers,
        ExecutionProvider,
    )

    # 1. Device name detection returns valid string
    gpu_name = get_gpu_device_name()
    assert isinstance(gpu_name, str)
    assert len(gpu_name) > 0

    # 2. CUDA session options
    c_opts, c_prov = get_cuda_session_options(device_id=0)
    assert c_opts is not None
    assert len(c_prov) >= 1
    assert c_prov[0]["device_id"] == "0"

    # 3. DirectML session options
    d_opts, d_prov = get_dml_session_options(device_id=0)
    assert d_opts is not None
    assert len(d_prov) >= 1
    assert d_prov[0]["device_id"] == "0"

    # 4. Provider routing
    cpu_prov = get_optimal_providers(ExecutionProvider.CPU)
    assert cpu_prov == ["CPUExecutionProvider"]

    # When CUDA requested on machine with DML only, falls back gracefully to DML/CPU
    cuda_prov = get_optimal_providers(ExecutionProvider.CUDA)
    assert isinstance(cuda_prov, list)
    assert len(cuda_prov) >= 1
    assert "CPUExecutionProvider" in cuda_prov

    # 5. UI Hardware status label formatting
    def format_hw_label(provs: list[str]) -> str:
        if any("CUDA" in str(p) for p in provs):
            return f"GPU: CUDA ({gpu_name})"
        elif any("Dml" in str(p) for p in provs):
            return f"GPU: DirectML ({gpu_name})"
        elif any("CoreML" in str(p) for p in provs):
            return f"GPU: CoreML ({gpu_name})"
        return "CPU"

    assert format_hw_label(["CUDAExecutionProvider", "CPUExecutionProvider"]) == f"GPU: CUDA ({gpu_name})"
    assert format_hw_label(["DmlExecutionProvider", "CPUExecutionProvider"]) == f"GPU: DirectML ({gpu_name})"
    assert format_hw_label(["CPUExecutionProvider"]) == "CPU"
