"""Repository layer handling database operations and FAISS synchronization."""
from __future__ import annotations

import datetime
import os
import re
import secrets
import threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker, Session as DBSession, joinedload, selectinload

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from .models import (
    Base,
    PersonGroup,
    Person,
    FaceTemplate,
    Camera,
    Profile,
    Session,
    Track,
    RecognitionEvent,
    User,
    AuditLog,
)
from core.recognizer import FaceRecognizer

ph = PasswordHasher()


class DatabaseRepo:
    """Provides high-level database operations and FAISS synchronization."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            db_url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL") or "sqlite:///data/face_app.db"
        self.db_url = db_url
        # Make directory if sqlite file path
        if db_url.startswith("sqlite:///"):
            db_file = db_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)

        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 30.0}

        self.engine = create_engine(db_url, echo=False, connect_args=connect_args)

        if db_url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                try:
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA busy_timeout=30000")
                    cursor.close()
                except Exception:
                    pass

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=self.engine)
        self.init_db()

    def init_db(self) -> None:
        """Creates tables and populates default user and settings if not present."""
        Base.metadata.create_all(bind=self.engine)
        with self.SessionLocal() as session:
            # Check default admin user from environment
            admin_username = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
            admin = session.query(User).filter_by(username=admin_username).first()
            if not admin:
                env_password = os.environ.get("ADMIN_PASSWORD")
                if env_password:
                    admin_password = env_password
                else:
                    admin_password = secrets.token_urlsafe(16)
                    print(f"\n[SECURITY] Initial admin user created: username='{admin_username}', password='{admin_password}'\n")
                admin = User(username=admin_username, password_hash=ph.hash(admin_password), role="admin")
                session.add(admin)

            # Check default group
            default_grp = session.query(PersonGroup).filter_by(name="General").first()
            if not default_grp:
                default_grp = PersonGroup(name="General")
                session.add(default_grp)

            # Check default webcam camera
            cam = session.query(Camera).first()
            if not cam:
                cam = Camera(
                    name="Default Webcam",
                    source_type="webcam",
                    transport="usb",
                    uri="0",
                    location="Local",
                    enabled=True,
                )
                session.add(cam)

            session.commit()

    # --- Person & Enrollment Management ---

    def create_person(
        self,
        code: str,
        full_name: str,
        group_id: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Person:
        with self.SessionLocal() as session:
            person = Person(
                code=code,
                full_name=full_name,
                group_id=group_id,
                notes=notes,
                status="active",
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            return person

    def get_person(self, person_id: int) -> Optional[Person]:
        with self.SessionLocal() as session:
            return session.query(Person).filter_by(id=person_id, deleted_at=None).first()

    def get_person_by_code(self, code: str) -> Optional[Person]:
        with self.SessionLocal() as session:
            return session.query(Person).filter_by(code=code, deleted_at=None).first()

    def list_persons(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            query = session.query(Person).options(
                joinedload(Person.group),
                selectinload(Person.templates),
            )
            if not include_deleted:
                query = query.filter(Person.deleted_at == None)
            persons = query.all()
            return [
                {
                    "id": p.id,
                    "code": p.code,
                    "full_name": p.full_name,
                    "group_id": p.group_id,
                    "group_name": p.group.name if p.group else "",
                    "status": p.status,
                    "notes": p.notes,
                    "template_count": len([t for t in p.templates if t.is_active]),
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in persons
            ]

    def delete_person(self, person_id: int, soft: bool = True) -> bool:
        with self.SessionLocal() as session:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person:
                return False
            if soft:
                person.deleted_at = datetime.datetime.utcnow()
                person.status = "inactive"
            else:
                session.delete(person)
            session.commit()
            return True

    def enroll_face(
        self,
        code: str,
        full_name: str,
        group_name: str = "General",
        templates: List[Tuple[np.ndarray, float, Optional[np.ndarray]]] = None,  # (embedding, quality, crop_img)
        recognizer: Optional[FaceRecognizer] = None,
        snapshots_dir: str = "data/snapshots",
    ) -> Person:
        """Atomically saves person, face templates, saves image crops to disk, and updates FAISS."""
        os.makedirs(snapshots_dir, exist_ok=True)
        if templates is None:
            templates = []

        with self.SessionLocal() as session:
            # Find or create group
            grp = session.query(PersonGroup).filter_by(name=group_name).first()
            if not grp:
                grp = PersonGroup(name=group_name)
                session.add(grp)
                session.flush()

            # Find or create person
            person = session.query(Person).filter_by(code=code).first()
            if not person:
                person = Person(
                    code=code,
                    full_name=full_name,
                    group_id=grp.id,
                    status="active",
                )
                session.add(person)
                session.flush()
            else:
                person.full_name = full_name
                person.group_id = grp.id
                person.status = "active"
                person.deleted_at = None

            created_templates = []
            saved_files = []  # Track files for cleanup on failure
            try:
                for i, (emb, quality, crop_img) in enumerate(templates):
                    img_path = None
                    if crop_img is not None and crop_img.size > 0:
                        clean_code = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(person.code))
                        fname = f"enroll_{clean_code}_{int(datetime.datetime.utcnow().timestamp())}_{i}.jpg"
                        full_path = os.path.join(snapshots_dir, fname)
                        try:
                            ext = os.path.splitext(full_path)[1] or ".jpg"
                            ok, buf = cv2.imencode(ext, crop_img)
                            if ok:
                                with open(full_path, "wb") as f_out:
                                    f_out.write(buf.tobytes())
                                saved_files.append(full_path)
                                img_path = full_path
                        except Exception as write_err:
                            print(f"[DatabaseRepo] Could not write snapshot {full_path}: {write_err}")

                    # Convert float32 array to binary
                    emb_norm = emb / max(1e-6, np.linalg.norm(emb))
                    emb_blob = emb_norm.astype(np.float32).tobytes()

                    template = FaceTemplate(
                        person_id=person.id,
                        embedding=emb_blob,
                        model_name="edgeface",
                        model_version="xxs",
                        quality=float(quality),
                        image_path=img_path,
                        is_active=True,
                    )
                    session.add(template)
                    session.flush()
                    created_templates.append((template.id, emb_norm))

                session.commit()
            except Exception:
                session.rollback()
                # Clean up orphaned files
                for fpath in saved_files:
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
                raise

            session.refresh(person)

            # Update FAISS
            if recognizer is not None:
                for tid, emb_norm in created_templates:
                    recognizer.add_template(
                        emb_norm,
                        {
                            "person_id": person.id,
                            "code": person.code,
                            "full_name": person.full_name,
                            "template_id": tid,
                        },
                    )
                recognizer.save()

            return person

    def count_person_templates(self, person_id: int) -> int:
        """Returns the number of active face templates associated with a person."""
        with self.SessionLocal() as session:
            return session.query(FaceTemplate).filter_by(person_id=person_id, is_active=True).count()

    def add_template_to_person(
        self,
        person_id: int,
        embedding: np.ndarray,
        quality: float,
        crop_img: Optional[np.ndarray] = None,
        recognizer: Optional[FaceRecognizer] = None,
        snapshots_dir: str = "data/snapshots",
    ) -> Optional[int]:
        """Adds a single face template to an existing person, saves image crop, and updates FAISS."""
        os.makedirs(snapshots_dir, exist_ok=True)
        with self.SessionLocal() as session:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person or person.deleted_at is not None or person.status != "active":
                return None

            img_path = None
            if crop_img is not None and crop_img.size > 0:
                clean_code = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(person.code))
                fname = f"learn_{clean_code}_{int(datetime.datetime.utcnow().timestamp())}_{np.random.randint(1000, 9999)}.jpg"
                full_path = os.path.join(snapshots_dir, fname)
                try:
                    ext = os.path.splitext(full_path)[1] or ".jpg"
                    ok, buf = cv2.imencode(ext, crop_img)
                    if ok:
                        with open(full_path, "wb") as f_out:
                            f_out.write(buf.tobytes())
                        img_path = full_path
                except Exception as write_err:
                    print(f"[DatabaseRepo] Could not write learn snapshot {full_path}: {write_err}")

            emb_norm = embedding / max(1e-6, np.linalg.norm(embedding))
            emb_blob = emb_norm.astype(np.float32).tobytes()

            template = FaceTemplate(
                person_id=person.id,
                embedding=emb_blob,
                model_name="edgeface",
                model_version="xxs",
                quality=float(quality),
                image_path=img_path,
                is_active=True,
            )
            session.add(template)
            session.commit()
            tid = template.id

            if recognizer is not None:
                recognizer.add_template(
                    emb_norm,
                    {
                        "person_id": person.id,
                        "code": person.code,
                        "full_name": person.full_name,
                        "template_id": tid,
                    },
                )
                recognizer.save()

            return tid

    def sync_faiss(self, recognizer: FaceRecognizer) -> int:
        """Rebuilds the FAISS vector index from active templates in the database."""
        with self.SessionLocal() as session:
            # Query active person IDs to ensure recognizer knows which identities are valid
            active_persons = session.query(Person.id).filter(Person.deleted_at == None, Person.status == "active").all()
            active_pids = {p[0] for p in active_persons}
            recognizer.set_active_persons(active_pids)

            templates = (
                session.query(FaceTemplate)
                .options(joinedload(FaceTemplate.person))
                .join(Person)
                .filter(FaceTemplate.is_active == True, Person.deleted_at == None, Person.status == "active")
                .yield_per(500)
            )

            indexed_data: List[Tuple[np.ndarray, Dict[str, Any]]] = []
            for t in templates:
                emb = np.frombuffer(t.embedding, dtype=np.float32).copy()
                meta = {
                    "person_id": t.person.id,
                    "code": t.person.code,
                    "full_name": t.person.full_name,
                    "template_id": t.id,
                }
                indexed_data.append((emb, meta))

            recognizer.rebuild_index(indexed_data)
            return len(indexed_data)

    # --- Session & Track & Event Logging ---

    def start_session(self, camera_id: Optional[int] = None, profile_id: Optional[int] = None, config_snapshot: str = "") -> int:
        with self.SessionLocal() as session:
            sess = Session(
                camera_id=camera_id,
                profile_id=profile_id,
                started_at=datetime.datetime.utcnow(),
                config_snapshot=config_snapshot,
            )
            session.add(sess)
            session.commit()
            return sess.id

    def end_session(self, session_id: int) -> None:
        with self.SessionLocal() as session:
            sess = session.query(Session).filter_by(id=session_id).first()
            if sess:
                sess.ended_at = datetime.datetime.utcnow()
                session.commit()

    def record_track_event(
        self,
        session_id: int,
        track_no: int,
        person_id: Optional[int],
        similarity: float,
        quality: float,
        is_unknown: bool,
        bbox: str,
        snapshot_crop: Optional[np.ndarray] = None,
        snapshots_dir: str = "data/snapshots",
    ) -> None:
        """Logs identification change or track snapshot."""
        os.makedirs(snapshots_dir, exist_ok=True)
        img_path = None
        if snapshot_crop is not None and snapshot_crop.size > 0:
            fname = f"event_s{session_id}_t{track_no}_{int(datetime.datetime.utcnow().timestamp())}.jpg"
            img_path = os.path.join(snapshots_dir, fname)
            cv2.imwrite(img_path, snapshot_crop)

        with self.SessionLocal() as session:
            # Find or create track
            trk = session.query(Track).filter_by(session_id=session_id, track_no=track_no).first()
            if not trk:
                trk = Track(
                    session_id=session_id,
                    track_no=track_no,
                    person_id=person_id,
                    best_quality=quality,
                    best_snapshot=img_path,
                    first_seen=datetime.datetime.utcnow(),
                    last_seen=datetime.datetime.utcnow(),
                )
                session.add(trk)
                session.flush()
            else:
                trk.last_seen = datetime.datetime.utcnow()
                if person_id is not None:
                    trk.person_id = person_id
                if quality > (trk.best_quality or 0.0):
                    trk.best_quality = quality
                    if img_path:
                        trk.best_snapshot = img_path

            event = RecognitionEvent(
                track_id=trk.id,
                person_id=person_id,
                similarity=similarity,
                quality=quality,
                is_unknown=is_unknown,
                ts=datetime.datetime.utcnow(),
                bbox=bbox,
                snapshot_path=img_path,
            )
            session.add(event)
            session.commit()

    def list_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            events = (
                session.query(RecognitionEvent)
                .options(joinedload(RecognitionEvent.person))
                .order_by(desc(RecognitionEvent.ts))
                .limit(limit)
                .offset(offset)
                .all()
            )
            out = []
            for ev in events:
                p_name = ev.person.full_name if ev.person else ("Unknown" if ev.is_unknown else "Unassigned")
                p_code = ev.person.code if ev.person else ""
                out.append({
                    "id": ev.id,
                    "track_id": ev.track_id,
                    "person_id": ev.person_id,
                    "person_name": p_name,
                    "person_code": p_code,
                    "similarity": ev.similarity,
                    "quality": ev.quality,
                    "is_unknown": ev.is_unknown,
                    "ts": ev.ts.isoformat() if ev.ts else "",
                    "bbox": ev.bbox,
                    "snapshot_path": ev.snapshot_path,
                })
            return out

    def reset_database(
        self,
        recognizer: Optional[FaceRecognizer] = None,
        clear_snapshots: bool = True,
        snapshots_dir: str = "data/snapshots",
    ) -> None:
        """Deletes all enrolled persons, face templates, tracks, and recognition events (wiping FAISS and snapshots)."""
        with self.SessionLocal() as session:
            session.query(RecognitionEvent).delete()
            session.query(Track).delete()
            session.query(FaceTemplate).delete()
            session.query(Person).delete()
            session.commit()

        if recognizer is not None:
            recognizer.reset()

        if clear_snapshots and os.path.exists(snapshots_dir):
            for fname in os.listdir(snapshots_dir):
                fpath = os.path.join(snapshots_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except OSError:
                    pass

    # --- User Authentication ---

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        with self.SessionLocal() as session:
            user = session.query(User).filter_by(username=username).first()
            if not user:
                return None
            try:
                ph.verify(user.password_hash, password)
                return user
            except VerifyMismatchError:
                return None

    def create_user(self, username: str, password: str, role: str = "operator") -> User:
        with self.SessionLocal() as session:
            user = User(
                username=username,
                password_hash=ph.hash(password),
                role=role,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user


_db_repo_instance: Optional[DatabaseRepo] = None
_db_repo_lock = threading.Lock()


def get_db_repo(db_url: Optional[str] = None) -> DatabaseRepo:
    global _db_repo_instance
    target_url = db_url or os.environ.get("DB_URL") or os.environ.get("DATABASE_URL") or "sqlite:///data/face_app.db"
    if _db_repo_instance is None or _db_repo_instance.db_url != target_url:
        with _db_repo_lock:
            if _db_repo_instance is None or _db_repo_instance.db_url != target_url:
                _db_repo_instance = DatabaseRepo(target_url)
    return _db_repo_instance
