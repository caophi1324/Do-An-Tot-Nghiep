"""FastAPI application providing MJPEG streaming, REST API, and web interface."""
from __future__ import annotations

import os
import time
import urllib.parse
import socket
import ipaddress
from contextlib import asynccontextmanager
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Body, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.config import AppConfig, TrackingMode, SystemMode
from core.runner import PipelineRunner
from core.sources import open_source
from db.repo import get_db_repo, DatabaseRepo
from db.models import User
from .auth import (
    authenticate_user,
    require_role,
    create_access_token,
    create_stream_token,
    verify_stream_token,
    TokenResponse,
    StreamTokenResponse,
    JWT_EXPIRE_MINUTES,
    get_current_user,
    get_jwt_secret,
    JWT_ALGORITHM,
)

_RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() not in ("false", "0", "off")
limiter = Limiter(key_func=get_remote_address, enabled=_RATE_LIMIT_ENABLED)

# Global instances initialized via lifespan or lazily on first access
config: Optional[AppConfig] = None
repo: Optional[DatabaseRepo] = None
runner: Optional[PipelineRunner] = None


def init_service_components() -> None:
    """Initializes models and database connections upon application startup."""
    global config, repo, runner
    if runner is not None:
        return

    # Validate JWT_SECRET on startup - rejects startup if missing or default
    get_jwt_secret()

    config_path = os.environ.get("CONFIG_PATH", "data/config.json")
    if os.path.exists(config_path):
        config = AppConfig.load_json(config_path)
    else:
        config = AppConfig()

    db_url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL") or config.storage.db_url
    repo = get_db_repo(db_url)
    runner = PipelineRunner(config)

    # Sync FAISS index
    try:
        repo.sync_faiss(runner.pipeline.recognizer)
    except Exception as e:
        print(f"[Service] Warning: FAISS sync failed: {e}")


def get_runner() -> PipelineRunner:
    global runner
    if runner is None:
        init_service_components()
    return runner


def get_repo() -> DatabaseRepo:
    global repo
    if repo is None:
        init_service_components()
    return repo


def get_config() -> AppConfig:
    global config
    if config is None:
        init_service_components()
    return config


def __getattr__(name: str) -> Any:
    if name == "runner":
        return get_runner()
    if name == "repo":
        return get_repo()
    if name == "config":
        return get_config()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate JWT_SECRET and load components/models
    init_service_components()
    yield
    # Shutdown: stop runner gracefully
    global runner
    if runner is not None and runner.is_running:
        runner.stop()


app = FastAPI(
    title="AI Face Recognition & Tracking API",
    description="Headless and Web REST API for Face Recognition and Tracking Pipeline",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: specify allowed origins explicitly when credentials are enabled
_ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth Endpoints ---
@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Tài khoản hoặc mật khẩu không chính xác",
        )
    token = create_access_token(user.username, user.role)
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
        role=user.role,
        username=user.username,
    )


@app.post("/auth/stream-token", response_model=StreamTokenResponse)
def issue_stream_token(user: User = Depends(require_role("viewer"))):
    """Generates a dedicated short-lived (~60s) token for streaming."""
    stream_token = create_stream_token(user.username)
    return StreamTokenResponse(stream_token=stream_token, expires_in=60)


# Pydantic Schemas
class SourceRequest(BaseModel):
    type: str  # webcam | droidcam | rtsp | file
    uri: str
    name: Optional[str] = "Camera 1"


class TrackingModeInput(str, PyEnum):
    person = "person"
    face = "face"


class SettingsRequest(BaseModel):
    match_thr: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_cutoff: Optional[float] = Field(None, ge=0.01, le=10.0)
    beta: Optional[float] = Field(None, ge=0.0, le=1.0)
    tracking_mode: Optional[TrackingModeInput] = None


@app.get("/health")
def health_check():
    r = get_runner()
    return {
        "status": "ok",
        "running": r.is_running,
        "recording": r.is_recording,
        "source_connected": r.source_connected,
        "fps": round(r.fps, 1),
        "latency_ms": round(r.latency_ms, 1),
        "total_templates": r.pipeline.recognizer.total_templates,
    }


STREAM_MAX_FRAMES: Optional[int] = None   # chỉ dùng cho test; production luôn None


def mjpeg_chunks(r, max_frames=None):
    sent = 0
    if hasattr(r, "subscribe_jpeg"):
        r.subscribe_jpeg()
    try:
        while max_frames is None or sent < max_frames:
            jpg = r.wait_next_jpeg(timeout=1.0)
            if jpg:
                sent += 1
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            else:
                time.sleep(0.04)
    except (ConnectionError, GeneratorExit):
        pass
    finally:
        if hasattr(r, "unsubscribe_jpeg"):
            r.unsubscribe_jpeg()


@app.get("/stream.mjpg")
def video_stream(token: str = Query(..., description="Short-lived 60s stream token")):
    """Streams MJPEG multipart frames. Requires ?token=<stream_token> issued via /auth/stream-token."""
    verify_stream_token(token)
    r = get_runner()
    return StreamingResponse(
        mjpeg_chunks(r, STREAM_MAX_FRAMES),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start")
@limiter.limit("10/minute")
def start_pipeline(request: Request, user: User = Depends(require_role("operator"))):
    r = get_runner()
    if not r.source:
        try:
            r.set_source(open_source("webcam", "0"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể mở webcam mặc định: {e}")

    r.start()
    return {"message": "Pipeline started", "running": True}


@app.post("/stop")
@limiter.limit("10/minute")
def stop_pipeline(request: Request, user: User = Depends(require_role("operator"))):
    r = get_runner()
    r.stop()
    return {"message": "Pipeline stopped", "running": False}


@app.post("/sources")
def set_source(req: SourceRequest, user: User = Depends(require_role("operator"))):
    src_type = req.type.lower().strip()

    # 1. Validate file sources: check existence & directory containment via realpath + commonpath
    if src_type == "file":
        real_target = os.path.realpath(req.uri)
        if not os.path.isfile(real_target):
            raise HTTPException(status_code=400, detail=f"File không tồn tại trên hệ thống: {req.uri}")

        allowed_dirs = [
            os.path.realpath("data"),
            os.path.realpath("videos"),
            os.path.realpath("tests/VideoTest"),
        ]
        extra_allowed = os.environ.get("ALLOWED_MEDIA_DIRS")
        if extra_allowed:
            for ed in extra_allowed.split(","):
                if ed.strip():
                    allowed_dirs.append(os.path.realpath(ed.strip()))

        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                if os.path.commonpath([real_target, allowed_dir]) == allowed_dir:
                    is_allowed = True
                    break
            except ValueError:
                continue

        if not is_allowed:
            raise HTTPException(
                status_code=400,
                detail="File path không được phép. Chỉ cho phép tệp nằm trong thư mục data/ hoặc videos/",
            )

    # 2. Validate network stream sources (SSRF protection & scheme check)
    elif src_type in ("stream", "droidcam", "rtsp", "http"):
        parsed = urllib.parse.urlparse(req.uri)
        if parsed.scheme.lower() not in ("http", "https", "rtsp"):
            raise HTTPException(
                status_code=400,
                detail=f"Scheme '{parsed.scheme}' không được phép. Chỉ cho phép http, https, rtsp",
            )

        hostname = parsed.hostname
        if not hostname:
            raise HTTPException(status_code=400, detail="URL không hợp lệ: thiếu hostname")

        allowlist_env = os.environ.get("STREAM_HOST_ALLOWLIST", "").strip()
        allowlist = [h.strip().lower() for h in allowlist_env.split(",") if h.strip()]
        if hostname.lower() not in allowlist:
            try:
                addr_infos = socket.getaddrinfo(hostname, None)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Không thể phân giải host: {hostname}")

            for _, _, _, _, sockaddr in addr_infos:
                ip_str = sockaddr[0]
                try:
                    ip_obj = ipaddress.ip_address(ip_str)
                except ValueError:
                    continue
                if (
                    ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_unspecified
                    or ip_str == "169.254.169.254"
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Host '{hostname}' ({ip_str}) bị chặn để bảo vệ an ninh mạng (SSRF protection)",
                    )

    r = get_runner()
    try:
        new_src = open_source(req.type, req.uri)
        r.set_source(new_src)
        if not r.is_running:
            r.start()
        return {"message": f"Source updated to {req.type} ({req.uri})", "connected": new_src.is_connected()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể mở nguồn: {e}")


@app.get("/persons")
def list_persons(user: User = Depends(require_role("viewer"))):
    return get_repo().list_persons()


@app.delete("/persons/{person_id}")
def delete_person(person_id: int, user: User = Depends(require_role("admin"))):
    rep = get_repo()
    ok = rep.delete_person(person_id, soft=True)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy đối tượng")
    rep.sync_faiss(get_runner().pipeline.recognizer)
    return {"message": f"Đã xóa đối tượng ID {person_id}"}


@app.patch("/settings")
def update_settings(req: SettingsRequest, user: User = Depends(require_role("operator"))):
    cfg = get_config()
    if req.match_thr is not None:
        cfg.recognition.match_thr = req.match_thr
    if req.min_quality is not None:
        cfg.quality.min_quality = req.min_quality
    if req.min_cutoff is not None:
        cfg.smoothing.min_cutoff = req.min_cutoff
    if req.beta is not None:
        cfg.smoothing.beta = req.beta
    if req.tracking_mode is not None:
        cfg.tracking.mode = TrackingMode.PERSON if req.tracking_mode == TrackingModeInput.person else TrackingMode.FACE

    get_runner().update_config(cfg)
    config_path = os.environ.get("CONFIG_PATH", "data/config.json")
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    cfg.save_json(config_path)
    return {"message": "Settings updated", "config": cfg.to_dict()}


@app.get("/events")
def get_events(limit: int = Query(50, ge=1, le=200), user: User = Depends(require_role("viewer"))):
    return get_repo().list_events(limit=limit)


@app.get("/", response_class=HTMLResponse)
def web_dashboard():
    """Embedded web dashboard for browser access."""
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>AI Face Studio Web Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #16171f; color: #e0e0e0; margin: 0; padding: 20px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2d2f3d; padding-bottom: 15px; }
            .header h2 { margin: 0; color: #ffffff; }
            .container { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 20px; }
            .stream-box { background: #0d0e14; border: 1px solid #2d2f3d; border-radius: 8px; overflow: hidden; text-align: center; }
            .stream-box img { width: 100%; max-height: 520px; object-fit: contain; }
            .card { background: #1e1f29; border: 1px solid #2d2f3d; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
            .btn { background: #2b78e4; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-right: 8px; }
            .btn-danger { background: #d9383a; }
            .btn-success { background: #2ea043; }
            .btn:hover { opacity: 0.9; }
            .stat { font-size: 13px; color: #9da5b4; margin: 6px 0; }
            .stat b { color: #ffffff; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🎥 AI Face Recognition & Tracking Studio</h2>
            <div id="stats-header" class="stat">FPS: <b id="fps-val">0.0</b> | Latency: <b id="lat-val">0ms</b></div>
        </div>
        <div class="container">
            <div class="stream-box">
                <img id="mjpg-stream" src="" alt="Live Camera Stream (Đăng nhập để xem)" />
            </div>
            <div>
                <div class="card">
                    <h3>🔐 Xác thực (Authentication)</h3>
                    <input type="text" id="username" placeholder="Username" style="background:#0d0e14;color:#fff;border:1px solid #333;padding:6px;border-radius:4px;margin-bottom:6px;width:90%;"><br>
                    <input type="password" id="password" placeholder="Password" style="background:#0d0e14;color:#fff;border:1px solid #333;padding:6px;border-radius:4px;margin-bottom:6px;width:90%;"><br>
                    <button class="btn" onclick="doLogin()">🔑 Đăng nhập</button>
                    <div id="auth-status" style="margin-top:8px;font-size:12px;color:#2ea043;"></div>
                </div>
                <div class="card">
                    <h3>🕹 Điều khiển (Controls)</h3>
                    <button class="btn btn-success" onclick="controlPipeline('/start')">▶ Bắt đầu</button>
                    <button class="btn btn-danger" onclick="controlPipeline('/stop')">⏹ Dừng</button>
                </div>
                <div class="card">
                    <h3>📊 Trạng thái hệ thống</h3>
                    <div class="stat">Tổng mẫu khuôn mặt: <b id="total-templates">0</b></div>
                    <div class="stat">Nguồn kết nối: <b id="source-status">Đang kết nối...</b></div>
                </div>
            </div>
        </div>
        <script>
            let authToken = localStorage.getItem('access_token') || '';

            async function refreshStream() {
                if (!authToken) return;
                try {
                    const res = await fetch('/auth/stream-token', {
                        method: 'POST',
                        headers: { 'Authorization': 'Bearer ' + authToken }
                    });
                    if (res.ok) {
                        const data = await res.json();
                        document.getElementById('mjpg-stream').src = '/stream.mjpg?token=' + data.stream_token;
                    }
                } catch(e) {}
            }

            if (authToken) {
                document.getElementById('auth-status').innerText = '✓ Đã lưu token';
                refreshStream();
            }

            async function doLogin() {
                const u = document.getElementById('username').value;
                const p = document.getElementById('password').value;
                const formData = new URLSearchParams();
                formData.append('username', u);
                formData.append('password', p);
                try {
                    const res = await fetch('/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: formData
                    });
                    if (res.ok) {
                        const data = await res.json();
                        authToken = data.access_token;
                        localStorage.setItem('access_token', authToken);
                        document.getElementById('auth-status').innerText = '✓ Đăng nhập thành công (' + data.role + ')';
                        await refreshStream();
                    } else {
                        document.getElementById('auth-status').innerText = '✗ Sai tài khoản hoặc mật khẩu';
                    }
                } catch(e) {
                    document.getElementById('auth-status').innerText = '✗ Lỗi kết nối: ' + e;
                }
            }

            async function controlPipeline(endpoint) {
                if (!authToken) { alert('Vui lòng đăng nhập trước!'); return; }
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
                if (!res.ok) {
                    alert('Lỗi gọi API: ' + res.status);
                }
            }

            setInterval(() => {
                fetch('/health')
                    .then(r => r.json())
                    .then(d => {
                        document.getElementById('fps-val').innerText = d.fps;
                        document.getElementById('lat-val').innerText = d.latency_ms + 'ms';
                        document.getElementById('total-templates').innerText = d.total_templates;
                        document.getElementById('source-status').innerText = d.source_connected ? 'Đang hoạt động' : 'Chưa kết nối';
                    }).catch(()=>{});
            }, 1000);

            // Refresh stream token every 45 seconds to keep connection active
            setInterval(() => {
                refreshStream();
            }, 45000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("service.main:app", host="0.0.0.0", port=port, reload=True)
