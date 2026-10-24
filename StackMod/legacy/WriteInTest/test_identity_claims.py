#!/usr/bin/env python3
"""
test_identity_claims.py - Test suite for claim management and synchronization on clear / expiration.
Covers:
1. Track expiration releases claim in IdentityGallery while retaining the identity.
2. clear_all_ids / face_tracker.clear() deletes manual_lock identities, preserves disk identities, releases all claims, and resets current_identity_id.
3. Key R / tracker.reset() preserves identities in gallery and only releases claim.
4. After clear, a new track with the reused track ID (e.g. 1) can claim an identity normally.
5. Deleting a track via FaceDBProxy.__delitem__ releases its claim.
6. (a) Đường thật: face_tracker.clear() rồi tracker.reset().
7. (b) PersonIDTracker.remove_track(tid, reason='deleted') nhả claim và dọn swap_votes.
8. (c) Test không treo (timeout 5s): face_tracker.update vs clear().
9. (c) Test không treo (timeout 5s): tracker.update vs reset().
10. (d/4) LOST face relock moves claim from old track to new vid with log [IDENTITY] claim_moved from=<old> to=<vid>.
11. (d/5) _maybe_enroll_identity checks claim result and logs WARNING [IDENTITY] claim_failed id=<..> track=<..> on False.
"""

import sys
import os
import time
import threading
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import logging

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

import tracking
import person_id_tracker
from config import settings
from tracking import TrackingManager
from person_id_tracker import PersonIDTracker
from identity_gallery import IdentityGallery
from face_utils import FaceGallery


class MockRecognizer:
    def __init__(self, dim=512):
        self.dim = dim

    def match(self, f1, f2, mode=0):
        v1 = np.asarray(f1, dtype=np.float32).flatten()
        v2 = np.asarray(f2, dtype=np.float32).flatten()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))


class MockDetector:
    pass


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class TestIdentityClaims(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dim = 512
        self.vecA = np.random.randn(self.dim).astype(np.float32)
        self.vecA /= np.linalg.norm(self.vecA)

        self.vecB = np.random.randn(self.dim).astype(np.float32)
        self.vecB /= np.linalg.norm(self.vecB)

        self.recognizer = MockRecognizer(dim=self.dim)
        self.detector = MockDetector()

        self.gallery = IdentityGallery(dim=self.dim)

        self.orig_recognition = dict(settings.recognition)
        settings.recognition["identity_enabled"] = True

        # Tránh ghi vào id_events.csv thật khi chạy test: patch trực tiếp trên các module đang dùng
        self.patch_person_id_logger = patch.object(person_id_tracker.id_logger, 'log_event', return_value=None)
        self.patch_person_id_logger.start()
        self.patch_tracking_id_logger = patch.object(tracking.id_logger, 'log_event', return_value=None)
        self.patch_tracking_id_logger.start()

        # Bắt log identity và loại bỏ FileHandler để không ghi vào logs/identity.log thật
        self.log_handler = LogCaptureHandler()
        self.id_logger = logging.getLogger("identity")
        self.orig_log_level = self.id_logger.level
        self.id_logger.setLevel(logging.INFO)
        self.removed_handlers = []
        for h in list(self.id_logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.id_logger.removeHandler(h)
                self.removed_handlers.append(h)
        self.id_logger.addHandler(self.log_handler)

    def tearDown(self):
        self.id_logger.removeHandler(self.log_handler)
        for h in self.removed_handlers:
            self.id_logger.addHandler(h)
        self.id_logger.setLevel(self.orig_log_level)
        self.patch_person_id_logger.stop()
        self.patch_tracking_id_logger.stop()
        settings.recognition.clear()
        settings.recognition.update(self.orig_recognition)

    def test_01_track_expired_releases_claim_and_retains_identity(self):
        """Track quá hạn (lost > max_lost) phải bị xóa khỏi tracked_faces,
        claim trong identity_gallery được nhả (release_track), nhưng identity vẫn tồn tại."""
        self.gallery.enroll("P1", [self.vecA], label="ID10", source="manual_lock")
        self.gallery.claim("P1", track_id=10)
        self.assertEqual(self.gallery.get("P1")["claim"], 10)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, max_lost=3, identity_gallery=self.gallery)
        face_tracker.tracked_faces[10] = {
            "bbox": [10, 10, 50, 50],
            "lost": 3,
            "t": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "gallery": FaceGallery(),
        }

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.log_handler.records.clear()
        face_tracker.update([], dummy_frame)

        self.assertNotIn(10, face_tracker.tracked_faces)
        self.assertIsNone(self.gallery.get("P1")["claim"])
        self.assertIsNotNone(self.gallery.get("P1"))

        release_logs = [r for r in self.log_handler.records if "[IDENTITY] release track=10 reason=expired" in r]
        self.assertTrue(len(release_logs) >= 1)

    def test_02_clear_all_ids_clears_manual_lock_keeps_disk_releases_all_claims(self):
        """clear_all_ids / face_tracker.clear():
        - Xóa identity nguồn manual_lock
        - Giữ nguyên identity nguồn disk
        - Nhả tất cả mọi claim
        - Reset current_identity_id của tracker và face_tracker."""
        self.gallery.enroll("P1", [self.vecA], label="ID1", source="manual_lock")
        self.gallery.enroll("DISK_TARGET", [self.vecB], label="Boss", source="disk")

        self.gallery.claim("P1", track_id=1)
        self.gallery.claim("DISK_TARGET", track_id=2)

        self.assertEqual(self.gallery.get("P1")["claim"], 1)
        self.assertEqual(self.gallery.get("DISK_TARGET")["claim"], 2)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        face_tracker.tracked_faces[1] = {"bbox": [0, 0, 10, 10], "gallery": FaceGallery(), "lost": 0}
        face_tracker.tracked_faces[2] = {"bbox": [20, 20, 30, 30], "gallery": FaceGallery(), "lost": 0}
        face_tracker.current_identity_id = "P1"

        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)
        face_tracker.tracker = mgr
        mgr.locked_face_id = 1
        mgr.current_identity_id = "P1"

        self.log_handler.records.clear()
        mgr.clear_all_ids()

        self.assertIsNone(self.gallery.get("P1"))
        disk_data = self.gallery.get("DISK_TARGET")
        self.assertIsNotNone(disk_data)
        self.assertEqual(disk_data["source"], "disk")
        self.assertIsNone(disk_data["claim"])

        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertIsNone(mgr.locked_face_id)
        self.assertEqual(face_tracker.next_id, 1)
        self.assertEqual(len(face_tracker.tracked_faces), 0)

        cleared_logs = [r for r in self.log_handler.records if "[IDENTITY] gallery_cleared n=1" in r]
        self.assertTrue(len(cleared_logs) >= 1)

    def test_03_reset_r_keeps_identity_only_releases_claim(self):
        """Nút R / tracker.reset():
        - Giữ nguyên identity trong gallery
        - Chỉ nhả claim của track đang lock
        - Reset current_identity_id về None."""
        self.gallery.enroll("P1", [self.vecA], label="ID1", source="manual_lock")
        self.gallery.claim("P1", track_id=1)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)
        mgr.locked_face_id = 1
        mgr.current_identity_id = "P1"
        face_tracker.current_identity_id = "P1"

        self.log_handler.records.clear()
        mgr.reset()

        self.assertIsNotNone(self.gallery.get("P1"))
        self.assertIsNone(self.gallery.get("P1")["claim"])
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertIsNone(mgr.locked_face_id)

        release_logs = [r for r in self.log_handler.records if "[IDENTITY] release track=1 reason=clear" in r]
        self.assertTrue(len(release_logs) >= 1)

    def test_04_after_clear_new_track_with_reused_id_can_claim_identity(self):
        """Sau khi clear, track mới có ID trùng số cũ (ID 1) vẫn claim được identity mới bình thường."""
        self.gallery.enroll("P1", [self.vecA], label="ID1", source="manual_lock")
        ok_claim1 = self.gallery.claim("P1", track_id=1)
        self.assertTrue(ok_claim1)
        self.assertEqual(self.gallery.get("P1")["claim"], 1)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        mgr.clear_all_ids()

        self.assertIsNone(self.gallery.get("P1"))
        self.assertEqual(face_tracker.next_id, 1)

        new_vec = np.random.randn(self.dim).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        self.gallery.enroll("P1_new", [new_vec], label="ID1_new", source="manual_lock")

        ok_claim2 = self.gallery.claim("P1_new", track_id=1)
        self.assertTrue(ok_claim2)
        self.assertEqual(self.gallery.get("P1_new")["claim"], 1)

    def test_05_facedb_proxy_delitem_releases_claim(self):
        """Xóa một track qua FaceDBProxy.__delitem__ phải nhả claim của track đó."""
        self.gallery.enroll("P5", [self.vecA], label="ID5", source="manual_lock")
        self.gallery.claim("P5", track_id=5)
        self.assertEqual(self.gallery.get("P5")["claim"], 5)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        face_tracker.tracked_faces[5] = {"bbox": [0, 0, 10, 10], "gallery": FaceGallery(), "lost": 0}

        self.log_handler.records.clear()
        del face_tracker.face_db[5]

        self.assertNotIn(5, face_tracker.tracked_faces)
        self.assertIsNone(self.gallery.get("P5")["claim"])
        self.assertIsNotNone(self.gallery.get("P5"))

        release_logs = [r for r in self.log_handler.records if "[IDENTITY] release track=5 reason=clear" in r]
        self.assertTrue(len(release_logs) >= 1)

    def test_06_real_ui_path_face_tracker_clear_then_tracker_reset(self):
        """Đường thật: face_tracker.clear() rồi tracker.reset() (như MainWindow.clear_all_ids)."""
        self.gallery.enroll("P1", [self.vecA], label="ID10", source="manual_lock")
        self.gallery.enroll("DISK_1", [self.vecB], label="Boss", source="disk")
        self.gallery.claim("P1", track_id=10)
        self.gallery.claim("DISK_1", track_id=20)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        face_tracker.tracked_faces[10] = {"bbox": [0, 0, 10, 10], "gallery": FaceGallery(), "lost": 0}
        face_tracker.tracked_faces[20] = {"bbox": [20, 20, 30, 30], "gallery": FaceGallery(), "lost": 0}
        face_tracker.current_identity_id = "P1"

        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)
        face_tracker.tracker = mgr
        mgr.locked_face_id = 10
        mgr.current_identity_id = "P1"

        self.log_handler.records.clear()
        face_tracker.clear()
        mgr.reset()

        self.assertIsNone(self.gallery.get("P1"))
        self.assertIsNotNone(self.gallery.get("DISK_1"))
        self.assertIsNone(self.gallery.get("DISK_1")["claim"])
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertIsNone(mgr.locked_face_id)
        self.assertEqual(face_tracker.next_id, 1)
        self.assertEqual(len(face_tracker.tracked_faces), 0)

        cleared_logs = [r for r in self.log_handler.records if "[IDENTITY] gallery_cleared n=1" in r]
        self.assertTrue(len(cleared_logs) >= 1)

    def test_07_remove_track_releases_claim_and_cleans_swap_votes(self):
        """PersonIDTracker.remove_track(tid, reason='deleted'):
        Nhả claim, xóa khỏi tracked_faces, và xóa mọi swap_votes chứa tid trong khóa tuple (cur, best)."""
        self.gallery.enroll("P5", [self.vecA], label="ID5", source="manual_lock")
        self.gallery.claim("P5", track_id=5)
        self.assertEqual(self.gallery.get("P5")["claim"], 5)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        fg = FaceGallery()
        fg.add(self.vecA)
        face_tracker.tracked_faces[5] = {"bbox": [0, 0, 10, 10], "gallery": fg, "lost": 0}
        face_tracker.swap_votes[(5, 7)] = 2
        face_tracker.swap_votes[(3, 5)] = 1
        face_tracker.swap_votes[(8, 9)] = 3

        self.log_handler.records.clear()
        face_tracker.remove_track(5, reason="deleted")

        self.assertNotIn(5, face_tracker.tracked_faces)
        self.assertIsNone(self.gallery.get("P5")["claim"])
        self.assertNotIn((5, 7), face_tracker.swap_votes)
        self.assertNotIn((3, 5), face_tracker.swap_votes)
        self.assertEqual(face_tracker.swap_votes.get((8, 9)), 3)

        release_logs = [r for r in self.log_handler.records if "[IDENTITY] release track=5 reason=deleted" in r]
        self.assertTrue(len(release_logs) >= 1)

    def test_08_no_deadlock_face_tracker_update_vs_clear(self):
        """Kiểm tra không deadlock giữa face_tracker.update (giữ rec_lock) và face_tracker.clear()."""
        rec_lock = threading.RLock()
        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        face_tracker.tracked_faces[1] = {
            "bbox": [10, 10, 50, 50], "lost": 0, "t": 0.0, "vx": 0.0, "vy": 0.0, "gallery": FaceGallery()
        }

        stop_flag = threading.Event()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def worker_update():
            while not stop_flag.is_set():
                with rec_lock:
                    face_tracker.update([[10, 10, 50, 50]], dummy_frame, rec_lock=rec_lock)
                    time.sleep(0.001)

        def worker_clear():
            for _ in range(5):
                face_tracker.clear()
                time.sleep(0.002)

        t1 = threading.Thread(target=worker_update)
        t2 = threading.Thread(target=worker_clear)

        t1.start()
        t2.start()

        t2.join(timeout=5.0)
        self.assertFalse(t2.is_alive(), "worker_clear bị treo/deadlock!")

        stop_flag.set()
        t1.join(timeout=5.0)
        self.assertFalse(t1.is_alive(), "worker_update bị treo/deadlock!")

    def test_09_no_deadlock_tracker_update_vs_reset(self):
        """Kiểm tra không deadlock giữa tracker.update (giữ rec_lock) và tracker.reset()."""
        rec_lock = threading.RLock()
        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(
            self.detector, self.recognizer, face_tracker=face_tracker,
            identity_gallery=self.gallery, rec_lock=rec_lock
        )
        mgr.state = "TRACKING"
        mgr.locked_face_id = 1
        mgr.locked_bbox = [10, 10, 50, 50]
        mgr._raw_locked_bbox = [10, 10, 50, 50]

        stop_flag = threading.Event()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def worker_update():
            while not stop_flag.is_set():
                with rec_lock:
                    mgr.update(dummy_frame, [[10, 10, 50, 50, 1]], (100, 100))
                    time.sleep(0.001)

        def worker_reset():
            for _ in range(5):
                mgr.reset()
                time.sleep(0.002)

        t1 = threading.Thread(target=worker_update)
        t2 = threading.Thread(target=worker_reset)

        t1.start()
        t2.start()

        t2.join(timeout=5.0)
        self.assertFalse(t2.is_alive(), "worker_reset bị treo/deadlock!")

        stop_flag.set()
        t1.join(timeout=5.0)
        self.assertFalse(t1.is_alive(), "worker_update bị treo/deadlock!")

    def test_10_lost_face_relock_moves_claim_and_logs(self):
        """Nhánh LOST 'Relock bằng khuôn mặt':
        Trước khi gán locked_face_id=vid, release claim của old_id rồi claim current_identity_id cho vid;
        log [IDENTITY] claim_moved from=<old> to=<vid>."""
        self.gallery.enroll("P1", [self.vecA], label="ID3", source="manual_lock")
        self.gallery.claim("P1", track_id=3)
        self.assertEqual(self.gallery.get("P1")["claim"], 3)

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)
        mgr.state = "LOST"
        mgr.locked_face_id = 3
        mgr.current_identity_id = "P1"
        face_tracker.current_identity_id = "P1"

        mgr.gallery.add(self.vecA, source="init")

        settings.recognition["face_threshold"] = 0.50
        settings.recognition["reid_margin"] = 0.05
        settings.recognition["recognition_interval_frames"] = 1

        self.log_handler.records.clear()

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [[10, 10, 50, 50, 7]]  # Track 7 xuất hiện

        with patch("tracking.extract_feat", return_value=(self.vecA, None)):
            # Frame 1: vote 1
            mgr.update(dummy_frame, dets, (100, 100))
            self.assertEqual(mgr.relock_votes.get(7), 1)

            # Frame 2: vote 2 -> trigger face relock
            mgr.update(dummy_frame, dets, (100, 100))

        self.assertEqual(mgr.locked_face_id, 7)
        self.assertEqual(mgr.state, "TRACKING")

        # Claim chuyển từ 3 sang 7
        self.assertEqual(self.gallery.get("P1")["claim"], 7)

        moved_logs = [r for r in self.log_handler.records if "[IDENTITY] claim_moved from=3 to=7" in r]
        self.assertTrue(len(moved_logs) >= 1, f"Logs: {self.log_handler.records}")

    def test_11_maybe_enroll_identity_claim_failed_logs_warning(self):
        """_maybe_enroll_identity: kiểm tra kết quả identity_gallery.claim;
        False thì log WARNING [IDENTITY] claim_failed id=<..> track=<..>."""
        class DummyProfile:
            def __init__(self, templates):
                self.fixed_templates = templates

        face_tracker = PersonIDTracker(self.detector, self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(self.detector, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)
        mgr.locked_face_id = 9

        # Tạo 3 vector thỏa mãn select_diverse (sim < 0.92) và medoid_sim (sim >= 0.40)
        v1 = np.random.randn(self.dim).astype(np.float32)
        v1 -= np.dot(self.vecA, v1) * self.vecA
        v1 /= np.linalg.norm(v1)

        t1 = self.vecA
        t2 = self.vecA + 0.5 * v1
        t2 /= np.linalg.norm(t2)
        t3 = self.vecA - 0.5 * v1
        t3 /= np.linalg.norm(t3)
        mgr.profile = DummyProfile([t1, t2, t3])
        mgr.profile_status = "ready"

        settings.recognition["identity_enabled"] = True
        settings.recognition["identity_medoid_sim"] = 0.40
        settings.recognition["profile_min_templates"] = 3

        # 1. Nhánh ENROLL: gallery rỗng, claim trả về False
        self.log_handler.records.clear()
        with patch.object(self.gallery, "claim", return_value=False):
            mgr._maybe_enroll_identity(source="manual")

        failed_enroll_logs = [r for r in self.log_handler.records if "[IDENTITY] claim_failed id=P1 track=9" in r]
        self.assertTrue(len(failed_enroll_logs) >= 1, f"Logs: {self.log_handler.records}")

        # 2. Nhánh MERGE: đã có P1 trong gallery, claim trả về False
        self.log_handler.records.clear()
        with patch.object(self.gallery, "claim", return_value=False):
            mgr._maybe_enroll_identity(source="manual")

        failed_merge_logs = [r for r in self.log_handler.records if "[IDENTITY] claim_failed id=P1 track=9" in r]
        self.assertTrue(len(failed_merge_logs) >= 1, f"Logs: {self.log_handler.records}")


if __name__ == "__main__":
    unittest.main()
