"""Regression and bug reproduction tests for Core, DB, Runner, Sources, UI, and Service.

Each test is documented with 'BUG: <mô tả>'.
Failing tests identifying existing known bugs in current code are marked with:
@pytest.mark.xfail(strict=True, reason="bug chưa sửa")
so the test suite runs clean while documenting exact reproduction steps.
"""
from __future__ import annotations

import os
import threading
import time
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from core.config import (
    AppConfig,
    RecognitionConfig,
    SmoothingConfig,
    TrackingConfig,
    TrackingMode,
    StorageConfig,
)
from core.tracker import TrackState, TrackManager
from core.pipeline import Pipeline, TrackResult
from core.runner import PipelineRunner
from core.sources import VideoSource, WebcamSource
from db.models import Person, FaceTemplate
from db.repo import DatabaseRepo
from tests.fakes import FakeSource, FakeRecognizer, build_fake_pipeline


# ==============================================================================
# 10. TRACKER TESTS
# ==============================================================================

def test_bug_tracker_confirmed_identity_drops_on_all_unknown():
    """BUG: Danh tính đã xác nhận phải rớt về Unknown khi cửa sổ bình chọn toàn phiếu Unknown."""
    recog_cfg = RecognitionConfig(vote_window=5, vote_min=3)
    smooth_cfg = SmoothingConfig()
    state = TrackState(track_id=1, recog_cfg=recog_cfg, smooth_cfg=smooth_cfg)

    # 1. Confirm identity initially as Person 1
    for _ in range(3):
        state.vote(pid=1, name="Person 1", sim=0.85)
    assert state.person_id == 1
    assert state.is_unknown is False

    # 2. Receive 5 consecutive Unknown votes (exceeding vote_window)
    for _ in range(5):
        state.vote(pid=None, name="Unknown", sim=0.20)

    # Expected behavior: identity should drop back to Unknown
    assert state.is_unknown is True, "Confirmed identity should drop to Unknown when all votes in window are Unknown"
    assert state.person_id is None, "person_id should be cleared when dropped to Unknown"


def test_tracker_hysteresis_drops_confirmed_identity_on_competing_votes():
    """Unit test item 3: Hysteresis drops confirmed identity when competing/unknown votes exceed threshold or similarity drops."""
    recog_cfg = RecognitionConfig(vote_window=5, vote_min=3, hysteresis_thr=0.60)
    smooth_cfg = SmoothingConfig()
    state = TrackState(track_id=1, recog_cfg=recog_cfg, smooth_cfg=smooth_cfg)

    # Confirm identity initially as Person 10
    for _ in range(3):
        state.vote(pid=10, name="Target", sim=0.80)
    assert state.person_id == 10
    assert not state.is_unknown

    # Low similarity votes below hysteresis threshold (0.50 < 0.60)
    for _ in range(5):
        state.vote(pid=10, name="Target", sim=0.50)
    assert state.person_id is None, "Identity should drop to Unknown when similarity drops below hysteresis_thr"
    assert state.is_unknown is True


def test_bug_tracker_one_to_one_identity_no_flicker_next_frame():
    """BUG: Ép 1-1 identity không được làm nhãn bị nhấp nháy lặp lại ở frame kế tiếp."""
    recog_cfg = RecognitionConfig(vote_window=5, vote_min=2)
    pipe = Pipeline(AppConfig())

    # Setup two tracks both having votes for Person 1
    state1 = TrackState(track_id=1, recog_cfg=recog_cfg, smooth_cfg=SmoothingConfig())
    state2 = TrackState(track_id=2, recog_cfg=recog_cfg, smooth_cfg=SmoothingConfig())
    pipe.tracker.states[1] = state1
    pipe.tracker.states[2] = state2

    state1.vote(pid=1, name="Person 1", sim=0.90)
    state1.vote(pid=1, name="Person 1", sim=0.90)
    state2.vote(pid=1, name="Person 1", sim=0.75)
    state2.vote(pid=1, name="Person 1", sim=0.75)

    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Mock detector results to return two tracks
    res1 = TrackResult(track_id=1, raw_bbox=np.array([10, 10, 50, 50]), smoothed_bbox=np.array([10, 10, 50, 50]), person_id=1, person_name="Person 1", similarity=0.90, quality=0.8, is_unknown=False)
    res2 = TrackResult(track_id=2, raw_bbox=np.array([60, 60, 100, 100]), smoothed_bbox=np.array([60, 60, 100, 100]), person_id=1, person_name="Person 1", similarity=0.75, quality=0.8, is_unknown=False)

    # Frame 1: 1-to-1 constraint demotes track 2
    pipe._enforce_exclusive_identities_and_locking([res1, res2])
    # Frame 2: verify track 2 does not retain old votes leading to immediate re-assignment
    state2.vote(pid=None, name="Unknown", sim=0.2)
    assert len([p for p, _, _ in state2.votes if p == 1]) == 0, "Demoted track should not retain competing votes"


def test_exclusive_identities_shared_helper_demotes_and_cleans_votes():
    """Unit test item 4: Shared helper demotes duplicate identity to Unknown and cleans votes."""
    pipe = Pipeline(AppConfig())
    state1 = TrackState(1, pipe.cfg.recognition, pipe.cfg.smoothing)
    state2 = TrackState(2, pipe.cfg.recognition, pipe.cfg.smoothing)
    pipe.tracker.states[1] = state1
    pipe.tracker.states[2] = state2

    state1.vote(5, "User5", 0.95)
    state2.vote(5, "User5", 0.70)

    res1 = TrackResult(1, np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10]), person_id=5, person_name="User5", similarity=0.95, is_unknown=False)
    res2 = TrackResult(2, np.array([20, 20, 30, 30]), np.array([20, 20, 30, 30]), person_id=5, person_name="User5", similarity=0.70, is_unknown=False)

    out = pipe._enforce_exclusive_identities_and_locking([res1, res2])
    r1 = next(r for r in out if r.track_id == 1)
    r2 = next(r for r in out if r.track_id == 2)

    assert r1.person_id == 5
    assert r2.person_id is None
    assert r2.is_unknown is True
    assert len([p for p, _, _ in state2.votes if p == 5]) == 0


# ==============================================================================
# 11. PIPELINE TESTS
# ==============================================================================

def test_bug_pipeline_tracking_mode_switch_resets_tracker():
    """BUG: Đổi tracking mode trong pipeline phải reset tracker."""
    cfg = AppConfig()
    cfg.tracking.mode = TrackingMode.PERSON
    pipe = Pipeline(cfg)

    # Add a mock active track state
    pipe.tracker.states[10] = TrackState(10, cfg.recognition, cfg.smoothing)
    assert len(pipe.tracker.states) == 1

    # Switch to FACE mode
    new_cfg = AppConfig()
    new_cfg.tracking.mode = TrackingMode.FACE
    pipe.update_config(new_cfg)

    assert len(pipe.tracker.states) == 0, "Tracker states must be cleared when switching tracking mode"


def test_pipeline_tracking_mode_switch_clears_locked_target_and_tracker():
    """Unit test item 1: Switching tracking mode clears locked target and tracker states."""
    cfg = AppConfig()
    cfg.tracking.mode = TrackingMode.PERSON
    pipe = Pipeline(cfg)
    pipe.lock_target(track_id=42, person_id=7)
    pipe.tracker.states[42] = TrackState(42, cfg.recognition, cfg.smoothing)

    assert pipe.locked_track_id == 42
    assert pipe.locked_person_id == 7
    assert len(pipe.tracker.states) == 1

    # Switch to FACE mode
    new_cfg = AppConfig()
    new_cfg.tracking.mode = TrackingMode.FACE
    pipe.update_config(new_cfg)

    assert pipe.locked_track_id is None
    assert pipe.locked_person_id is None
    assert len(pipe.tracker.states) == 0


def test_bug_pipeline_smooth_params_applied_to_running_tracks():
    """BUG: Thay đổi min_cutoff/beta phải áp dụng cho track đang chạy."""
    cfg = AppConfig()
    cfg.smoothing.min_cutoff = 1.0
    cfg.smoothing.beta = 0.01
    pipe = Pipeline(cfg)

    state = TrackState(track_id=1, recog_cfg=cfg.recognition, smooth_cfg=cfg.smoothing)
    pipe.tracker.states[1] = state

    # Dynamically update smoothing config
    new_cfg = AppConfig()
    new_cfg.smoothing.min_cutoff = 4.5
    new_cfg.smoothing.beta = 0.25
    pipe.update_config(new_cfg)

    # Check active track's smoother
    assert state.smoother.x_smoother.min_cutoff == 4.5, "Active smoother min_cutoff should be updated"
    assert state.smoother.x_smoother.beta == 0.25, "Active smoother beta should be updated"


# ==============================================================================
# 12. RUNNER TESTS
# ==============================================================================

def test_bug_runner_duplicate_timestamp_ignored():
    """BUG: PipelineRunner không được xử lý frame có timestamp trùng với frame trước."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Source with duplicate timestamp 1.0 twice
    source = FakeSource([(frame, 1.0), (frame, 1.0), (frame, 2.0)])
    runner = PipelineRunner(AppConfig(), source=source)

    processed_timestamps = []
    original_process = runner.pipeline.process

    def mock_process(f, timestamp=None):
        processed_timestamps.append(timestamp)
        return original_process(f, timestamp)

    runner.pipeline.process = mock_process
    runner.start()
    time.sleep(0.3)
    runner.stop()

    assert processed_timestamps.count(1.0) == 1, "Duplicate frame with same timestamp should be ignored"


def test_bug_runner_stop_recording_during_loop_does_not_crash(tmp_path):
    """BUG: stop_recording giữa lúc loop chạy không làm thread chết."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = FakeSource([(frame, float(i)) for i in range(100)], loop=True)
    cfg = AppConfig()
    cfg.storage.recordings_dir = str(tmp_path)
    runner = PipelineRunner(cfg, source=source)

    runner.start()
    time.sleep(0.1)
    runner.start_recording()
    time.sleep(0.1)
    runner.stop_recording()
    time.sleep(0.1)

    assert runner.is_running is True
    assert runner._thread is not None and runner._thread.is_alive()
    runner.stop()


def test_bug_runner_set_source_maintains_single_worker_thread():
    """BUG: set_source khi đang chạy chỉ còn đúng 1 worker thread."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source1 = FakeSource([(frame, float(i)) for i in range(50)], loop=True)
    source2 = FakeSource([(frame, float(i)) for i in range(50)], loop=True)
    runner = PipelineRunner(AppConfig(), source=source1)

    runner.start()
    time.sleep(0.1)
    runner.set_source(source2)
    time.sleep(0.1)

    assert runner.is_running is True
    assert runner._thread is not None and runner._thread.is_alive()
    runner.stop()


def test_bug_runner_jpeg_subscribers_flag_cleared_on_disconnect():
    """BUG: Sau khi ngắt client stream, _has_jpeg_subscribers về False."""
    runner = PipelineRunner(AppConfig())
    assert runner._has_jpeg_subscribers is False

    # 1. Subscriber connects
    runner.subscribe_jpeg()
    assert runner._has_jpeg_subscribers is True

    # 2. After client stream disconnects, runner reverts _has_jpeg_subscribers to False
    runner.unsubscribe_jpeg()
    assert runner._has_jpeg_subscribers is False, "_has_jpeg_subscribers should be False when no clients are subscribed"

    # 3. Test with mjpeg_chunks generator try/finally lifecycle
    from service.main import mjpeg_chunks
    runner.wait_next_jpeg = MagicMock(return_value=b"fake_frame")
    gen = mjpeg_chunks(runner, max_frames=1)
    list(gen)
    assert runner._has_jpeg_subscribers is False


# ==============================================================================
# 13. REPO TESTS
# ==============================================================================

@pytest.mark.xfail(strict=True, reason="bug chưa sửa: xoá rồi enroll lại cùng code hồi sinh template cũ")
def test_bug_repo_re_enroll_same_code_does_not_resurrect_deleted_templates(tmp_path):
    """BUG: Xoá rồi enroll lại cùng code không hồi sinh template cũ."""
    repo = DatabaseRepo(f"sqlite:///{tmp_path}/test_repo_resurrect.db")
    v1 = np.ones(512, dtype=np.float32)

    # 1. Enroll initial person
    p1 = repo.enroll_face(code="EMP10", full_name="User Old", templates=[(v1, 0.9, None)], snapshots_dir=str(tmp_path))
    assert repo.count_person_templates(p1.id) == 1

    # 2. Delete person
    repo.delete_person(p1.id, soft=True)

    # 3. Re-enroll same code with new template
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    p2 = repo.enroll_face(code="EMP10", full_name="User New", templates=[(v2, 0.85, None)], snapshots_dir=str(tmp_path))

    # Old templates should NOT be resurrected
    assert repo.count_person_templates(p2.id) == 1, "Deleted templates should not be resurrected upon re-enrollment"


def test_bug_repo_add_template_rejects_deleted_person(tmp_path):
    """BUG: add_template_to_person từ chối person đã xoá."""
    repo = DatabaseRepo(f"sqlite:///{tmp_path}/test_repo_del.db")
    v1 = np.ones(512, dtype=np.float32)

    p = repo.enroll_face(code="EMP20", full_name="To Delete", templates=[(v1, 0.9, None)], snapshots_dir=str(tmp_path))
    repo.delete_person(p.id, soft=True)

    # Attempt to add template to deleted person
    v2 = np.zeros(512, dtype=np.float32)
    tid = repo.add_template_to_person(p.id, v2, 0.9, snapshots_dir=str(tmp_path))
    assert tid is None, "Should reject adding template to deleted person"


@pytest.mark.xfail(strict=True, reason="bug chưa sửa: code chứa ../ không ghi file ra ngoài snapshots_dir")
def test_bug_repo_code_with_path_traversal_does_not_escape_snapshots_dir(tmp_path):
    """BUG: Code chứa '../' không ghi file ra ngoài snapshots_dir."""
    repo = DatabaseRepo(f"sqlite:///{tmp_path}/test_repo_trav.db")
    snaps_dir = tmp_path / "snaps"
    (snaps_dir / "enroll_sub").mkdir(parents=True, exist_ok=True)

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    v = np.ones(512, dtype=np.float32)

    # Attempt enrollment with path traversal code: "sub/../../escaped"
    repo.enroll_face(code="sub/../../escaped", full_name="Escaped", templates=[(v, 0.9, crop)], snapshots_dir=str(snaps_dir))

    # Expected: No file should be written outside snaps_dir
    parent_files = [f for f in tmp_path.iterdir() if f.is_file() and not f.name.endswith(".db")]
    assert len(parent_files) == 0, f"Image crop escaped snapshots_dir: {parent_files}"


# ==============================================================================
# 14. SOURCES TESTS
# ==============================================================================

def test_bug_sources_read_does_not_return_duplicate_stale_frame():
    """BUG: WebcamSource.read không trả frame cũ lần thứ hai khi chưa có frame mới."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Yield one frame then nothing
        mock_cap.read.side_effect = [(True, mock_frame), (False, None)]
        mock_cap_cls.return_value = mock_cap

        src = WebcamSource(device_index=0)
        time.sleep(0.1)

        ts1, f1 = src.read()
        assert f1 is not None

        # Second read before new frame arrives should return None
        ts2, f2 = src.read()
        src.stop()
        assert f2 is None, "Second read without a new captured frame should return None rather than duplicating stale frame"


# ==============================================================================
# 15. UI TESTS (@ui)
# ==============================================================================

@pytest.mark.ui
def test_bug_ui_tuning_dock_reset_defaults_matches_app_config_defaults(qapp):
    """BUG: TuningDock.reset_defaults khớp AppConfig mặc định."""
    from ui.docks.tuning_dock import TuningDock
    dock = TuningDock()
    dock.reset_defaults()

    default_cfg = AppConfig()
    dock_thr = dock.match_slider.value() / 100.0
    assert dock_thr == default_cfg.recognition.match_thr, f"TuningDock default ({dock_thr}) differs from AppConfig ({default_cfg.recognition.match_thr})"


@pytest.mark.ui
def test_bug_ui_sources_dock_reorder_does_not_emit_source_selected(qapp):
    """BUG: SourcesDock sắp xếp lại không phát source_selected."""
    from ui.docks.sources_dock import SourcesDock
    dock = SourcesDock()
    dock.add_source_item("Camera 2", "webcam", "1")

    emitted_sources = []
    dock.source_selected.connect(lambda t, u: emitted_sources.append((t, u)))

    # Move item down (reorder)
    dock._move_down()

    assert len(emitted_sources) == 0, "Reordering sources should not trigger active camera switch"


@pytest.mark.ui
@pytest.mark.xfail(strict=True, reason="bug chưa sửa: SettingsDialog._on_save không ghi storage path")
def test_bug_ui_settings_dialog_saves_storage_path(qapp):
    """BUG: SettingsDialog._on_save ghi storage path."""
    from ui.dialogs.settings_dialog import SettingsDialog
    cfg = AppConfig()
    recog = FakeRecognizer()
    repo = DatabaseRepo("sqlite:///:memory:")

    dialog = SettingsDialog(cfg, repo, recog)
    # Check if dialog has storage path controls
    assert hasattr(dialog, "edit_storage_path"), "SettingsDialog missing storage path configuration fields"


# ==============================================================================
# 16. SERVICE TESTS
# ==============================================================================

@pytest.mark.timeout(20)
def test_sources_rejects_loopback_ssrf(client, auth_headers):
    """BUG: POST /sources rejects SSRF loopback host 127.0.0.1."""
    res = client.post("/sources", json={"type": "stream", "uri": "http://127.0.0.1:8000/feed"}, headers=auth_headers)
    assert res.status_code == 400
    assert "SSRF" in res.json().get("detail", "")


@pytest.mark.timeout(20)
def test_sources_rejects_metadata_ip(client, auth_headers):
    """BUG: POST /sources rejects SSRF cloud metadata IP 169.254.169.254."""
    res = client.post("/sources", json={"type": "rtsp", "uri": "rtsp://169.254.169.254/h264"}, headers=auth_headers)
    assert res.status_code == 400
    assert "SSRF" in res.json().get("detail", "")


@pytest.mark.timeout(20)
def test_sources_rejects_file_scheme(client, auth_headers):
    """BUG: POST /sources rejects disallowed scheme file:///."""
    res = client.post("/sources", json={"type": "stream", "uri": "file:///etc/passwd"}, headers=auth_headers)
    assert res.status_code == 400


@pytest.mark.timeout(20)
def test_stream_requires_token(client):
    """BUG: GET /stream.mjpg rejects missing (422) or invalid (401) token."""
    res_no_tok = client.get("/stream.mjpg")
    assert res_no_tok.status_code == 422

    res_bad_tok = client.get("/stream.mjpg?token=invalid.jwt.token")
    assert res_bad_tok.status_code == 401


@pytest.mark.timeout(20)
def test_access_token_rejected_by_stream(client, auth_headers):
    """BUG: General access token must be rejected by /stream.mjpg (requires dedicated stream token)."""
    access_token = auth_headers["Authorization"].split()[1]
    res = client.get(f"/stream.mjpg?token={access_token}")
    assert res.status_code == 401


@pytest.mark.timeout(20)
def test_stream_token_rejected_by_api(client, auth_headers):
    """BUG: Short-lived stream token must not be accepted by regular API endpoints like /persons."""
    res_st = client.post("/auth/stream-token", headers=auth_headers)
    assert res_st.status_code == 200
    stream_tok = res_st.json()["stream_token"]

    res_persons = client.get("/persons", headers={"Authorization": f"Bearer {stream_tok}"})
    assert res_persons.status_code == 401


@pytest.mark.timeout(20)
def test_stream_yields_jpeg(client, auth_headers, monkeypatch):
    """BUG: GET /stream.mjpg streams multipart JPEG frames when provided valid stream token."""
    import service.main
    monkeypatch.setattr(service.main, "STREAM_MAX_FRAMES", 1)

    mock_r = MagicMock()
    mock_r.wait_next_jpeg.return_value = b"fake_frame_bytes"
    monkeypatch.setattr(service.main, "get_runner", lambda: mock_r)

    res_st = client.post("/auth/stream-token", headers=auth_headers)
    assert res_st.status_code == 200
    tok = res_st.json()["stream_token"]

    res = client.get(f"/stream.mjpg?token={tok}")
    assert res.status_code == 200
    assert b"fake_frame_bytes" in res.content


@pytest.mark.timeout(20)
def test_mjpeg_chunks_stops_at_max_frames():
    """BUG: mjpeg_chunks stops generator after max_frames limit is reached."""
    from service.main import mjpeg_chunks
    fake_runner = MagicMock()
    fake_runner.wait_next_jpeg.return_value = b"test_jpg"

    chunks = list(mjpeg_chunks(fake_runner, max_frames=2))
    assert len(chunks) == 2
    assert all(b"test_jpg" in c for c in chunks)


@pytest.mark.timeout(20)
def test_rate_limiter_login_429():
    """BUG: Verifies slowapi RateLimiter returns 429 when enabled and limit is exceeded."""
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    test_limiter = Limiter(key_func=get_remote_address, enabled=True)
    test_app = FastAPI()
    test_app.state.limiter = test_limiter
    test_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    test_app.add_middleware(SlowAPIMiddleware)

    @test_app.post("/test-login")
    @test_limiter.limit("2/minute")
    def login_endpoint(request: Request):
        return {"status": "ok"}

    c = TestClient(test_app)
    assert c.post("/test-login").status_code == 200
    assert c.post("/test-login").status_code == 200
    r3 = c.post("/test-login")
    assert r3.status_code == 429


# ==============================================================================
# 17. CONCURRENCY HARDENING & CORE SOURCES ACCEPTANCE TESTS
# ==============================================================================

def test_runner_stop_release_source_flag():
    """Item 5: PipelineRunner.stop(release_source=True) stops video source."""
    mock_src = MagicMock()
    runner = PipelineRunner(AppConfig(), source=mock_src)
    runner.stop(release_source=False)
    mock_src.stop.assert_not_called()

    runner.stop(release_source=True)
    mock_src.stop.assert_called_once()


def test_runner_lock_unlock_does_not_hold_runner_lock():
    """Item 6: lock_target / unlock_target does not hold runner._lock."""
    runner = PipelineRunner(AppConfig())
    with runner._lock:
        # If lock_target / unlock_target held runner._lock, it would deadlock here
        runner.lock_target(1, 100)
        assert runner.locked_target == (1, 100)
        runner.unlock_target()
        assert runner.locked_target == (None, None)


@pytest.mark.ui
def test_enroll_dialog_inference_lock_and_throttling(qapp):
    """Item 8: EnrollDialog throttles live frame eval and protects model calls with inference_lock."""
    from ui.dialogs.enroll_dialog import EnrollDialog
    lock = threading.RLock()
    mock_det = MagicMock()
    mock_det.detect_in_frame.return_value = []
    mock_recog = MagicMock()
    mock_q = MagicMock()
    mock_repo = MagicMock()

    dlg = EnrollDialog(
        repo=mock_repo,
        recognizer=mock_recog,
        face_det=mock_det,
        quality_est=mock_q,
        cfg=AppConfig(),
        inference_lock=lock,
    )
    assert dlg.inference_lock is lock

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dlg.set_live_frame(dummy_frame)
    time.sleep(0.05)
    call_count_1 = mock_det.detect_in_frame.call_count
    # Second immediate frame within 0.20s should be throttled
    dlg.set_live_frame(dummy_frame)
    assert mock_det.detect_in_frame.call_count == call_count_1


def test_stream_source_passes_opencv_timeout_options():
    """Item 9: StreamSource passes timeout properties directly into cv2.VideoCapture."""
    import cv2
    from core.sources import StreamSource
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        src = StreamSource("rtsp://camera.local/stream", reconnect_interval_s=10.0)
        time.sleep(0.1)
        src.stop()

        assert mock_cap_cls.called
        args, kwargs = mock_cap_cls.call_args
        assert "rtsp://camera.local/stream" in args
        params = args[2] if len(args) > 2 else []
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params
        assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in params
        assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


@pytest.mark.ui
def test_sources_dock_remove_blocks_signals(qapp):
    """Item 10: SourcesDock remove button blocks source_selected signal."""
    from ui.docks.sources_dock import SourcesDock
    dock = SourcesDock()
    dock.add_source_item("Camera 2", "webcam", "1")
    emitted = []
    dock.source_selected.connect(lambda t, u: emitted.append((t, u)))
    dock.list_widget.setCurrentRow(1)
    emitted.clear()
    dock._remove_source()
    assert len(emitted) == 0, "Removing source item must not emit source_selected"


def test_bug_face_bbox_smoothing_and_landmark_motion_propagation():
    """BUG: BBox của mặt và landmark 5 điểm phải mượt và di chuyển theo người trong các frame cadence skip."""
    from core.pipeline import Pipeline
    from core.config import AppConfig, TrackingMode
    from core.detectors import FaceDetectionResult

    cfg = AppConfig()
    cfg.tracking.mode = TrackingMode.PERSON
    pipe = Pipeline(cfg)

    # Mock detectors
    person_box_1 = np.array([100.0, 100.0, 300.0, 500.0, 0.9], dtype=np.float32)
    face_box_1 = np.array([160.0, 120.0, 240.0, 200.0], dtype=np.float32)
    landmarks_1 = np.array([
        [180.0, 145.0],
        [220.0, 145.0],
        [200.0, 165.0],
        [185.0, 185.0],
        [215.0, 185.0],
    ], dtype=np.float32)

    with patch.object(pipe.person_det, "detect") as mock_pdet, \
         patch.object(pipe.face_det, "detect_in_person_crop") as mock_fdet:

        # Frame 1: Person detected, face detected
        mock_pdet.return_value = [
            MagicMock(x1=100.0, y1=100.0, x2=300.0, y2=500.0, conf=0.9)
        ]
        mock_fdet.return_value = FaceDetectionResult(
            bbox=face_box_1,
            landmarks=landmarks_1,
            conf=0.95,
        )

        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        res1 = pipe.process(frame1, timestamp=1.0)
        assert len(res1) == 1
        r1 = res1[0]
        assert r1.face_bbox is not None
        assert r1.landmarks is not None
        initial_nose_x = r1.landmarks[2, 0]

        # Frame 2: Person walked right (+30px), cadence skips face detection (mock returns None)
        mock_pdet.return_value = [
            MagicMock(x1=130.0, y1=100.0, x2=330.0, y2=500.0, conf=0.9)
        ]
        mock_fdet.return_value = None

        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        res2 = pipe.process(frame2, timestamp=1.033)
        assert len(res2) == 1
        r2 = res2[0]

        # Face bbox must be tracked and smoothed
        assert r2.face_bbox is not None
        assert r2.face_bbox[0] > r1.face_bbox[0], "Face bbox must track moving person"

        # Landmarks must not be None and must NOT be frozen at frame 1 position
        assert r2.landmarks is not None
        assert r2.landmarks[2, 0] > initial_nose_x, "Landmarks must move with person, not freeze in place"


def test_bug_detect_in_person_crop_adds_context_margin():
    """BUG: detect_in_person_crop phải mở rộng biên crop để không cắt phạm vào trán/cằm làm hỏng 5-landmark."""
    from core.detectors import FaceDetector
    from core.config import FaceDetectionConfig

    cfg = FaceDetectionConfig()
    detector = FaceDetector(cfg)

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # Person box: y1=200, y2=800 (height 600), x1=300, x2=500 (width 200)
    person_box = [300, 200, 500, 800]

    with patch.object(detector.model, "detect", return_value=[]) as mock_detect:
        detector.detect_in_person_crop(frame, person_box)
        assert mock_detect.called
        crop_arg = mock_detect.call_args[0][0]
        # Crop height should be larger than upper 60% (360px) due to padding (pad_top=60, pad_bottom=30)
        # 600 * 0.65 + 60 + 30 = ~480px > 360px
        assert crop_arg.shape[0] > 360, "Crop height should include top/bottom padding context"
        # Crop width should be larger than 200px due to side padding (200 + 2*24 = 248px)
        assert crop_arg.shape[1] > 200, "Crop width should include horizontal padding context"

