"""SQLAlchemy ORM models matching project specifications."""
from __future__ import annotations

import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    LargeBinary,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PersonGroup(Base):
    __tablename__ = "person_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)

    persons = relationship("Person", back_populates="group", cascade="all, delete-orphan")


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), nullable=False, unique=True, index=True)
    full_name = Column(String(150), nullable=False)
    group_id = Column(Integer, ForeignKey("person_groups.id"), nullable=True)
    status = Column(String(20), nullable=False, default="active")  # active | blocked | inactive
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    group = relationship("PersonGroup", back_populates="persons")
    templates = relationship("FaceTemplate", back_populates="person", cascade="all, delete-orphan")
    tracks = relationship("Track", back_populates="person")
    events = relationship("RecognitionEvent", back_populates="person")


class FaceTemplate(Base):
    __tablename__ = "face_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # Float32 bytes (512 float32 = 2048 bytes)
    model_name = Column(String(50), nullable=False, default="edgeface")
    model_version = Column(String(50), nullable=False, default="xxs")
    quality = Column(Float, nullable=True)
    image_path = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="templates")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    source_type = Column(String(30), nullable=False)  # webcam | file | droidcam | rtsp
    transport = Column(String(30), nullable=True, default="usb")  # usb | mjpeg | rtsp
    uri = Column(String(500), nullable=False)
    location = Column(String(200), nullable=True)
    width = Column(Integer, nullable=True, default=1280)
    height = Column(Integer, nullable=True, default=720)
    blur_thr = Column(Float, nullable=True, default=0.2)
    q_low = Column(Float, nullable=True, default=0.3)
    match_thr_offset = Column(Float, nullable=True, default=0.0)
    enhance_mode = Column(String(30), nullable=True, default="face")
    enabled = Column(Boolean, nullable=False, default=True)

    sessions = relationship("Session", back_populates="camera")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    mode = Column(String(30), nullable=False, default="simple")
    config_json = Column(Text, nullable=False)

    sessions = relationship("Session", back_populates="profile")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    config_snapshot = Column(Text, nullable=True)

    camera = relationship("Camera", back_populates="sessions")
    profile = relationship("Profile", back_populates="sessions")
    tracks = relationship("Track", back_populates="session", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    track_no = Column(Integer, nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    best_quality = Column(Float, nullable=True, default=0.0)
    best_snapshot = Column(String(255), nullable=True)
    enhanced_ratio = Column(Float, nullable=True, default=0.0)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("session_id", "track_no", name="uq_session_track_no"),)

    session = relationship("Session", back_populates="tracks")
    person = relationship("Person", back_populates="tracks")
    events = relationship("RecognitionEvent", back_populates="track", cascade="all, delete-orphan")


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True)
    similarity = Column(Float, nullable=True)
    quality = Column(Float, nullable=True)
    is_unknown = Column(Boolean, nullable=False, default=False)
    enhanced = Column(Boolean, nullable=False, default=False)
    enhancer_version = Column(String(50), nullable=True)
    ts = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    bbox = Column(String(100), nullable=True)  # "x1,y1,x2,y2"
    snapshot_path = Column(String(255), nullable=True)

    track = relationship("Track", back_populates="events")
    person = relationship("Person", back_populates="events")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="operator")  # admin | operator | viewer
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    logs = relationship("AuditLog", back_populates="user")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    target = Column(String(255), nullable=True)
    ts = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="logs")
