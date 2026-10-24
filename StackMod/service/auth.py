"""JWT-based authentication and role-based access control for FastAPI endpoints."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import jwt

from db.repo import get_db_repo, DatabaseRepo
from db.models import User

# --- Configuration ---
INSECURE_SECRET_DEFAULTS = {
    "",
    "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32",
    "change-me-to-a-secure-random-secret",
}


def get_jwt_secret() -> str:
    """Retrieves and validates JWT_SECRET from environment. Rejects missing or default insecure values."""
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret or secret in INSECURE_SECRET_DEFAULTS or secret.startswith("CHANGE_ME") or secret.startswith("change-me"):
        raise RuntimeError(
            "JWT_SECRET is not configured or uses default insecure value. "
            "Please set a strong JWT_SECRET environment variable (e.g., using openssl rand -hex 32)."
        )
    return secret


JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))  # 8 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

ROLE_HIERARCHY = {
    "viewer": 1,
    "operator": 2,
    "admin": 3,
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: str
    username: str


class StreamTokenResponse(BaseModel):
    stream_token: str
    expires_in: int = 60  # seconds


def create_access_token(username: str, role: str) -> str:
    """Creates a JWT token with expiry and role claim."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_stream_token(username: str) -> str:
    """Creates a short-lived (~60s) single-purpose token exclusively for video streaming."""
    expire = datetime.now(timezone.utc) + timedelta(seconds=60)
    payload = {
        "sub": username,
        "type": "stream",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Validates credentials against database."""
    repo = get_db_repo()
    return repo.authenticate_user(username, password)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Decodes JWT and returns the authenticated User."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise credentials_exception
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exception

    repo = get_db_repo()
    with repo.SessionLocal() as session:
        user = session.query(User).filter_by(username=username).first()
        if user is None:
            raise credentials_exception
        return user


def verify_stream_token(token: str) -> str:
    """Verifies a dedicated short-lived stream token and ensures the user still exists in DB."""
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        if not username or token_type != "stream":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token stream không hợp lệ hoặc sai loại token",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token stream đã hết hạn (60s). Vui lòng yêu cầu token mới.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token stream không hợp lệ",
        )

    repo = get_db_repo()
    with repo.SessionLocal() as session:
        user = session.query(User).filter_by(username=username).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Người dùng không còn tồn tại trong hệ thống",
            )
    return username


def require_role(min_role: str):
    """Dependency checking that the authenticated user meets minimum role requirement."""
    min_level = ROLE_HIERARCHY.get(min_role.lower(), 1)

    def role_checker(user: User = Depends(get_current_user)) -> User:
        user_role = (user.role or "viewer").lower()
        user_level = ROLE_HIERARCHY.get(user_role, 1)
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Quyền hạn không đủ. Yêu cầu tối thiểu vai trò: '{min_role}'",
            )
        return user

    return role_checker
