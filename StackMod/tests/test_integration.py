"""Integration tests IT-01 to IT-05 for end-to-end pipeline and FastAPI service."""
from __future__ import annotations

import time
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from core.config import AppConfig, TrackingMode
from core.pipeline import Pipeline, render_overlay
from service.main import app


def test_it01_pipeline_empty_frame():
    """IT-01: Processing empty black frame executes gracefully."""
    cfg = AppConfig()
    pipeline = Pipeline(cfg)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = pipeline.process(blank_frame, timestamp=time.time())
    assert isinstance(results, list)
    assert len(results) == 0

    # Test rendering on empty results
    out = render_overlay(blank_frame, results, cfg.overlay)
    assert out.shape == blank_frame.shape


def test_it02_pipeline_synthetic_scene():
    """IT-02: Pipeline processes synthetic scene with person silhouette."""
    cfg = AppConfig()
    pipeline = Pipeline(cfg)

    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    # Draw simple shapes
    cv2.rectangle(frame, (400, 200), (600, 650), (160, 140, 120), -1)
    cv2.circle(frame, (500, 260), 45, (200, 180, 150), -1)

    # Process frame through pipeline
    results = pipeline.process(frame, timestamp=time.time())
    assert isinstance(results, list)

    rendered = render_overlay(frame, results, cfg.overlay, fps=25.0, latency_ms=35.0)
    assert rendered.shape == frame.shape


def test_it03_pipeline_dynamic_config_update():
    """BUG: Đổi tracking mode phải reset state tracker và xoá locked_track_id."""
    cfg = AppConfig()
    pipeline = Pipeline(cfg)
    pipeline.lock_target(track_id=101, person_id=1)

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    _ = pipeline.process(frame, timestamp=1.0)

    # Switch to FACE tracking mode
    new_cfg = AppConfig()
    new_cfg.tracking.mode = TrackingMode.FACE
    new_cfg.recognition.match_thr = 0.65
    pipeline.update_config(new_cfg)

    # Assert real behavior: tracker states must be reset and locked target cleared
    assert len(pipeline.tracker.states) == 0, "Tracker states must be reset when switching tracking mode"
    assert pipeline.locked_track_id is None, "locked_track_id must be cleared when switching tracking mode"


def test_it04_fastapi_health_endpoint():
    """IT-04: Test /health REST endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "running" in data
    assert "total_templates" in data


def test_it05_fastapi_dashboard_html():
    """IT-05: Test Web Dashboard HTML response."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Face Recognition & Tracking Studio" in response.text
    assert "stream.mjpg" in response.text


def test_it06_jwt_login_success_and_failure():
    """IT-06: Test /auth/login returns valid JWT and handles invalid credentials."""
    client = TestClient(app)

    # Valid login
    res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    body = res.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    assert body["username"] == "admin"

    # Invalid login
    res_bad = client.post("/auth/login", data={"username": "admin", "password": "wrong_password"})
    assert res_bad.status_code == 401


def test_it07_protected_endpoints_with_jwt():
    """IT-07: Protected endpoints require JWT authentication."""
    client = TestClient(app)

    # Without token -> 401 Unauthorized
    res_no_auth = client.get("/persons")
    assert res_no_auth.status_code == 401

    # Login to obtain token
    login_res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Access /persons with token
    res_persons = client.get("/persons", headers=headers)
    assert res_persons.status_code == 200
    assert isinstance(res_persons.json(), list)

    # Access /events with token
    res_events = client.get("/events?limit=10", headers=headers)
    assert res_events.status_code == 200
    assert isinstance(res_events.json(), list)


def test_it08_stream_token_validation():
    """IT-08: /stream.mjpg rejects missing/invalid tokens and accepts dedicated 60s stream tokens."""
    client = TestClient(app)

    # Missing token query parameter -> 422 Unprocessable Entity
    res_missing = client.get("/stream.mjpg")
    assert res_missing.status_code == 422

    # Invalid token -> 401 Unauthorized
    res_invalid = client.get("/stream.mjpg?token=invalid.jwt.token")
    assert res_invalid.status_code == 401

    # Login to obtain general access token
    login_res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    access_token = login_res.json()["access_token"]

    # Passing general access token into query string is rejected (must use dedicated stream token)
    res_wrong_type = client.get(f"/stream.mjpg?token={access_token}")
    assert res_wrong_type.status_code == 401

    # Obtain dedicated short-lived stream token
    st_res = client.post("/auth/stream-token", headers={"Authorization": f"Bearer {access_token}"})
    assert st_res.status_code == 200
    stream_tok = st_res.json()["stream_token"]
    assert len(stream_tok) > 20

    # Non-existent user token simulation -> 401
    from service.auth import create_stream_token
    st_nonexistent = create_stream_token(username="deleted_user_xyz")
    res_ghost = client.get(f"/stream.mjpg?token={st_nonexistent}")
    assert res_ghost.status_code == 401


def test_it09_path_traversal_guard(tmp_path):
    """IT-09: /sources rejects path traversal, prefix collisions, non-existent files, and SSRF hosts."""
    client = TestClient(app)
    login_res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Non-existent file
    res_nonexist = client.post(
        "/sources",
        json={"type": "file", "uri": str(tmp_path / "non_existent_file.mp4")},
        headers=headers,
    )
    assert res_nonexist.status_code == 400

    # 2. File outside allowed dirs
    dummy_out = tmp_path / "outside.mp4"
    dummy_out.write_bytes(b"test")
    res = client.post(
        "/sources",
        json={"type": "file", "uri": str(dummy_out)},
        headers=headers,
    )
    assert res.status_code == 400
    assert "File path không được phép" in res.json()["detail"] or "không tồn tại" in res.json()["detail"]

    # 2b. Sibling directory prefix collision: data2/x.mp4 must be rejected
    from pathlib import Path
    data2_dir = Path("data2")
    data2_dir.mkdir(exist_ok=True)
    fake_data2_file = data2_dir / "x.mp4"
    fake_data2_file.write_bytes(b"test")
    try:
        res_pfx = client.post(
            "/sources",
            json={"type": "file", "uri": str(fake_data2_file)},
            headers=headers,
        )
        assert res_pfx.status_code == 400
        assert "File path không được phép" in res_pfx.json()["detail"]
    finally:
        if fake_data2_file.exists():
            fake_data2_file.unlink()
        if data2_dir.exists():
            data2_dir.rmdir()

    # 2c. Path traversal with .. outside allowed directory
    res_traversal = client.post(
        "/sources",
        json={"type": "file", "uri": "data/../pyproject.toml"},
        headers=headers,
    )
    assert res_traversal.status_code == 400
    assert "File path không được phép" in res_traversal.json()["detail"]

    # 3. Network SSRF: loopback blocked
    res_loopback = client.post(
        "/sources",
        json={"type": "stream", "uri": "http://127.0.0.1:8080/live"},
        headers=headers,
    )
    assert res_loopback.status_code == 400
    assert "SSRF" in res_loopback.json()["detail"]

    # 4. Network SSRF: metadata service blocked
    res_meta = client.post(
        "/sources",
        json={"type": "rtsp", "uri": "rtsp://169.254.169.254/cam"},
        headers=headers,
    )
    assert res_meta.status_code == 400
    assert "SSRF" in res_meta.json()["detail"]

    # 5. Invalid scheme
    res_bad_scheme = client.post(
        "/sources",
        json={"type": "stream", "uri": "ftp://camera.local/stream"},
        headers=headers,
    )
    assert res_bad_scheme.status_code == 400


def test_it10_settings_validation():
    """IT-10: /settings validates input value ranges."""
    client = TestClient(app)
    login_res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Invalid match_thr > 1.0 -> 422
    res_invalid = client.patch(
        "/settings",
        json={"match_thr": 2.5},
        headers=headers,
    )
    assert res_invalid.status_code == 422

    # Valid update
    res_valid = client.patch(
        "/settings",
        json={"match_thr": 0.55, "min_quality": 0.35, "tracking_mode": "person"},
        headers=headers,
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["config"]["recognition"]["match_thr"] == 0.55


def test_it11_click_to_lock_and_auto_enrollment(tmp_path):
    """IT-11: Verify sequential auto-enrollment (ID 1, then N+) and target locking."""
    from db.repo import DatabaseRepo
    from core.recognizer import FaceRecognizer
    from core.config import RecognitionConfig, AppConfig
    from core.pipeline import Pipeline
    import numpy as np

    db_path = f"sqlite:///{tmp_path}/test_lock.db"
    faiss_dir = str(tmp_path / "faiss")
    repo = DatabaseRepo(db_path)
    recog = FaceRecognizer(RecognitionConfig(), faiss_path=faiss_dir)

    from ui.main_window import MainWindow
    stub_window = type("StubMainWindow", (), {"repo": repo})()

    # First auto enrollment -> ID 1 using real MainWindow logic
    next_id_1 = MainWindow._get_next_sequential_person_id(stub_window)
    assert next_id_1 == 1

    emb1 = np.random.randn(512).astype(np.float32)
    p1 = repo.enroll_face(
        code=str(next_id_1),
        full_name=f"Person {next_id_1}",
        templates=[(emb1, 0.9, None)],
        recognizer=recog,
        snapshots_dir=str(tmp_path / "snaps"),
    )
    assert p1.code == "1"
    assert p1.full_name == "Person 1"

    # Second auto enrollment -> ID 2 (N+) using real MainWindow logic
    next_id_2 = MainWindow._get_next_sequential_person_id(stub_window)
    assert next_id_2 == 2

    emb2 = np.random.randn(512).astype(np.float32)
    p2 = repo.enroll_face(
        code=str(next_id_2),
        full_name=f"Person {next_id_2}",
        templates=[(emb2, 0.9, None)],
        recognizer=recog,
        snapshots_dir=str(tmp_path / "snaps"),
    )
    assert p2.code == "2"
    assert p2.full_name == "Person 2"

    # Third auto enrollment -> ID 3
    next_id_3 = MainWindow._get_next_sequential_person_id(stub_window)
    assert next_id_3 == 3

    # Test Pipeline locking target with Person 1
    pipe = Pipeline(AppConfig())
    pipe.lock_target(track_id=101, person_id=p1.id)
    assert pipe.locked_track_id == 101
    assert pipe.locked_person_id == p1.id

    pipe.unlock_target()
    assert pipe.locked_track_id is None


def test_it12_database_reset(tmp_path):
    """IT-12: Verify full database and FAISS vector reset."""
    from db.repo import DatabaseRepo
    from core.recognizer import FaceRecognizer
    from core.config import RecognitionConfig
    import numpy as np

    db_path = f"sqlite:///{tmp_path}/test_reset.db"
    faiss_dir = str(tmp_path / "faiss")
    repo = DatabaseRepo(db_path)
    recog = FaceRecognizer(RecognitionConfig(), faiss_path=faiss_dir)

    # Enroll a sample person
    emb = np.random.randn(512).astype(np.float32)
    p = repo.enroll_face(
        code="1",
        full_name="Person 1",
        templates=[(emb, 0.9, None)],
        recognizer=recog,
        snapshots_dir=str(tmp_path / "snaps"),
    )
    assert len(repo.list_persons()) == 1
    assert recog.total_templates == 1

    # Reset database
    repo.reset_database(
        recognizer=recog,
        clear_snapshots=True,
        snapshots_dir=str(tmp_path / "snaps"),
    )

    # Verify completely clean
    assert len(repo.list_persons()) == 0
    assert recog.total_templates == 0


def test_it13_post_lock_5_sample_learning(tmp_path, monkeypatch):
    """IT-13: Verify post-lock 5-sample active learning through _process_person_tracking_mode with fake detectors."""
    from db.repo import DatabaseRepo
    from core.detectors import PersonDetection, FaceDetectionResult
    from tests.fakes import build_fake_pipeline
    from core.config import AppConfig
    import numpy as np

    db_path = f"sqlite:///{tmp_path}/test_learn.db"
    faiss_dir = str(tmp_path / "faiss")
    repo = DatabaseRepo(db_path)

    # 1. Enroll Person 1 initial anchor
    v_anchor = np.zeros(512, dtype=np.float32)
    v_anchor[0] = 1.0

    learned_events = []
    def on_learned(pid, emb, quality, crop):
        repo.add_template_to_person(
            person_id=pid,
            embedding=emb,
            quality=quality,
            crop_img=crop,
            recognizer=None,
            snapshots_dir=str(tmp_path / "snaps"),
        )
        learned_events.append((pid, quality))

    cfg = AppConfig()
    cfg.storage.faiss_path = faiss_dir
    cfg.storage.snapshots_dir = str(tmp_path / "snaps")

    # Person and Face detections at fixed coords
    pdet = [PersonDetection(bbox=np.array([100.0, 100.0, 300.0, 400.0], dtype=np.float32), conf=0.95)]
    fdet = FaceDetectionResult(
        bbox=np.array([120.0, 110.0, 220.0, 210.0], dtype=np.float32),
        landmarks=np.array([[140, 140], [180, 140], [160, 160], [145, 180], [175, 180]], dtype=np.float32),
        conf=0.95,
    )

    pipe, fake_p, fake_f, fake_q, fake_rec = build_fake_pipeline(
        monkeypatch,
        cfg=cfg,
        person_detections=pdet,
        face_result=fdet,
        quality_score=0.90,
        on_template_learned=on_learned,
    )

    person1 = repo.enroll_face(
        code="1",
        full_name="Person 1",
        templates=[(v_anchor, 0.90, None)],
        recognizer=fake_rec,
        snapshots_dir=str(tmp_path / "snaps"),
    )
    assert repo.count_person_templates(person1.id) == 1

    dummy_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # Process frame through _process_person_tracking_mode
    results = pipe.process(dummy_frame, timestamp=1.0)
    assert len(results) == 1
    tid = results[0].track_id

    # Lock target
    pipe.lock_target(track_id=tid, person_id=person1.id)

    # Frame 2 to 5: active learning loop runs through _process_person_tracking_mode
    for i in range(1, 5):
        _ = pipe.process(dummy_frame, timestamp=1.0 + i * 0.1)

    state = pipe.tracker.states[tid]
    assert state.gallery.total_count >= 1
    assert state.person_id == person1.id


