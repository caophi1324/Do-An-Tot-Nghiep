"""Pytest configuration and global isolation fixtures."""
from __future__ import annotations

import os
import shutil
import tempfile
import pytest

# Pre-set safe test environment variables before any module import occurs
_SESSION_TMP_DIR = tempfile.mkdtemp(prefix="face_app_test_iso_")
_TEST_DB_PATH = os.path.join(_SESSION_TMP_DIR, "test_face_app.db")
_TEST_CONFIG_PATH = os.path.join(_SESSION_TMP_DIR, "test_config.json")
_TEST_FAISS_DIR = os.path.join(_SESSION_TMP_DIR, "test_faiss")
_TEST_SNAPS_DIR = os.path.join(_SESSION_TMP_DIR, "test_snapshots")

os.environ["JWT_SECRET"] = "pytest-secret-for-testing-only-32-chars-minimum-abcdef123"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "admin123"
os.environ["DB_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"
os.environ["CONFIG_PATH"] = _TEST_CONFIG_PATH
os.environ["FAISS_PATH"] = _TEST_FAISS_DIR
os.environ["SNAPSHOTS_DIR"] = _TEST_SNAPS_DIR
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session", autouse=True)
def cleanup_tmp_session():
    """Ensures no test ever writes to the production data/ directory and cleans up session tmpdir."""
    yield
    # Cleanup session temporary directory
    if os.path.exists(_SESSION_TMP_DIR):
        try:
            shutil.rmtree(_SESSION_TMP_DIR, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture
def isolated_storage(tmp_path):
    """Provides an isolated StorageConfig rooted entirely in tmp_path."""
    from core.config import StorageConfig
    db_file = str(tmp_path / "face_app.db")
    return StorageConfig(
        db_url=f"sqlite:///{db_file}",
        faiss_path=str(tmp_path / "faiss"),
        snapshots_dir=str(tmp_path / "snapshots"),
        recordings_dir=str(tmp_path / "recordings"),
    )


@pytest.fixture(scope="session")
def qapp():
    """Provides a QApplication instance running offscreen for UI tests."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from service.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_headers(client):
    """Session-scoped auth headers logged in as admin once."""
    res = client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
