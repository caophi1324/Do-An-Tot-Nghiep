#!/usr/bin/env python3
"""
test_identity_enroll_hook.py - Test suite for IdentityGallery integration into TrackingManager & PersonIDTracker
Tests:
1. Auto lock is skipped from enrolling into IdentityGallery.
2. Manual lock meeting conditions enrolls new identity ("P1").
3. Re-locking same person merges into existing identity ("P1") without creating new identity ("P2").
4. Locking a different person creates new identity ("P2").
5. Inconsistent / low-medoid history skips enrolling.
6. Online learning branch enrolls when builder becomes ready in update().
7. Adaptive sync adds template to IdentityGallery under current_identity_id.
8. Target drop and reset clear current_identity_id and release claims.
"""

import sys
import os
import unittest
import numpy as np
import logging

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

from config import settings
from tracking import TrackingManager
from person_id_tracker import PersonIDTracker
from identity_gallery import IdentityGallery


class MockRecognizer:
    """Mock recognizer calculating cosine similarity."""
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


def make_diverse_cluster(base_vec, n=5, noise=0.35):
    """
    Tạo chùm mẫu vừa đủ nhất quán để ProfileBuilder chấp nhận (sim ~ 0.85 với base_vec),
    vừa đủ đa dạng (sim pairwise ~ 0.78 < min_gap_sim 0.92) để không bị lọc hết bởi select_diverse.
    """
    samples = []
    for _ in range(n):
        noise_dir = np.random.randn(*base_vec.shape).astype(np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        v = (1.0 - noise) * base_vec + noise * noise_dir
        samples.append((v / np.linalg.norm(v)).astype(np.float32))
    return samples


class TestIdentityEnrollHook(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.personA = np.random.randn(512).astype(np.float32)
        self.personA /= np.linalg.norm(self.personA)

        self.personB = np.random.randn(512).astype(np.float32)
        self.personB /= np.linalg.norm(self.personB)

        self.recognizer = MockRecognizer(dim=512)
        self.detector = MockDetector()

        self.orig_recognition = dict(settings.recognition)
        # Configure settings for identity & target profile
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_min_templates"] = 3
        settings.recognition["profile_target_templates"] = 5
        settings.recognition["face_threshold"] = 0.38
        settings.recognition["identity_enabled"] = True
        settings.recognition["identity_max_templates"] = 12
        settings.recognition["identity_min_gap_sim"] = 0.92
        settings.recognition["identity_min_fixed_sim"] = 0.45
        settings.recognition["identity_medoid_sim"] = 0.40
        settings.recognition["identity_merge_sim"] = 0.50

        self.gallery = IdentityGallery(
            min_gap_sim=0.92,
            max_templates=12,
            top_k=3,
            min_fixed_sim=0.45,
            dim=512,
        )

    def tearDown(self):
        settings.recognition.clear()
        settings.recognition.update(self.orig_recognition)

    def test_01_auto_lock_is_skipped(self):
        """Auto lock (source='auto') phai bi bo qua, khong enroll vao IdentityGallery."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        history = make_diverse_cluster(self.personA, n=5, noise=0.35)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history, source="auto")

        self.assertEqual(mgr.profile_status, "ready")
        self.assertEqual(len(self.gallery.list_identities()), 0)
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)

    def test_02_manual_lock_enrolls_new_identity(self):
        """Manual lock (source='manual') du dieu kien phai enroll thanh cong 'P1'."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        history = make_diverse_cluster(self.personA, n=5, noise=0.35)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history, source="manual")

        self.assertEqual(mgr.profile_status, "ready")
        identities = self.gallery.list_identities()
        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0]["id"], "P1")
        self.assertEqual(identities[0]["label"], "ID1")
        self.assertEqual(identities[0]["source"], "manual_lock")
        self.assertEqual(identities[0]["claim"], 1)

        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(face_tracker.current_identity_id, "P1")

    def test_03_relock_same_person_merges_without_new_id(self):
        """Khoa lai cung nguoi A voi track ID khac: phai gop vao 'P1', khong tao 'P2'."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Lan 1: Khoa track 1 (nguoi A) -> enroll P1
        history_a1 = make_diverse_cluster(self.personA, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=history_a1, source="manual")
        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(len(self.gallery.list_identities()), 1)

        # Reset / Mat track
        mgr.reset()
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(self.gallery.list_identities()[0]["claim"])

        # Lan 2: Khoa track 2 (cung nguoi A) -> merge vao P1
        history_a2 = make_diverse_cluster(self.personA, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [120, 120, 220, 320, 2], 640, 480, history=history_a2, source="manual")

        # Khong tao identity moi!
        identities = self.gallery.list_identities()
        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0]["id"], "P1")
        self.assertEqual(identities[0]["claim"], 2)
        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(face_tracker.current_identity_id, "P1")

    def test_04_lock_different_person_creates_p2(self):
        """Khoa nguoi B khac hoan toan nguoi A: tao identity moi 'P2'."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Khoa nguoi A -> P1
        hist_a = make_diverse_cluster(self.personA, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=hist_a)
        self.assertEqual(mgr.current_identity_id, "P1")

        # Reset
        mgr.reset()

        # Khoa nguoi B -> P2
        hist_b = make_diverse_cluster(self.personB, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 2], 640, 480, history=hist_b)
        self.assertEqual(mgr.current_identity_id, "P2")

        identities = self.gallery.list_identities()
        self.assertEqual(len(identities), 2)
        ids = {it["id"] for it in identities}
        self.assertEqual(ids, {"P1", "P2"})

    def test_05_skip_when_medoid_sim_too_low(self):
        """Lich su bi phan tan / lan tap chat khien medoid sim < 0.50 -> Bo qua enroll."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        # Tao tap mau phan tan xung quanh 3 vector truc giao
        v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0
        v3 = np.zeros(512, dtype=np.float32); v3[2] = 1.0
        scattered_history = [v1, v2, v3]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=scattered_history)

        self.assertEqual(len(self.gallery.list_identities()), 0)
        self.assertIsNone(mgr.current_identity_id)

    def test_06_online_learning_enrolls_when_builder_ready(self):
        """Khoa chua co lich su (learning), khi update tich luy du mau va build ok -> phai enroll."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=[], source="manual")

        self.assertEqual(mgr.profile_status, "learning")
        self.assertIsNone(mgr.current_identity_id)
        self.assertEqual(len(self.gallery.list_identities()), 0)

        # Nap mau vao builder
        cluster = make_diverse_cluster(self.personA, n=5, noise=0.35)
        for feat in cluster:
            mgr.builder.add(feat)

        self.assertTrue(mgr.builder.ready())
        prof, info = mgr.builder.build()
        self.assertEqual(info["reason"], "ok")
        mgr.profile = prof
        mgr.profile_status = "ready"
        mgr._maybe_enroll_identity(source=mgr.lock_source)

        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(len(self.gallery.list_identities()), 1)

    def test_07_adaptive_sync_to_identity_gallery(self):
        """Khi maybe_adapt thanh cong, mau moi phai duoc add vao IdentityGallery cua ID dang khoa."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hist = make_diverse_cluster(self.personA, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=hist)
        self.assertEqual(mgr.current_identity_id, "P1")

        initial_count = self.gallery.get("P1")["n_templates"]

        # Tao mot mau moi tuong dong tot voi fixed templates (sim ~ 0.80) nhung da dang (sim < 0.92)
        new_feat = make_diverse_cluster(self.personA, n=1, noise=0.30)[0]
        # Verify sim voi fixed templates
        fixed = self.gallery.get("P1")["templates"]
        sims_to_fixed = [float(np.dot(new_feat, f)) for f in fixed]
        self.assertGreaterEqual(max(sims_to_fixed), 0.45)
        self.assertLess(max(sims_to_fixed), 0.92)

        # Goi add_template qua hook adaptive
        ok, reason = self.gallery.add_template(mgr.current_identity_id, new_feat)
        self.assertTrue(ok, f"add_template failed: {reason}")
        self.assertEqual(self.gallery.get("P1")["n_templates"], initial_count + 1)

    def test_08_reset_and_drop_cleans_identity_and_claims(self):
        """Drop target hoac reset phai xoa current_identity_id va nhả claim."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hist = make_diverse_cluster(self.personA, n=5, noise=0.35)
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=hist)

        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(self.gallery.list_identities()[0]["claim"], 1)

        # Test _drop_target
        mgr._drop_target("test_drop")
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertIsNone(self.gallery.list_identities()[0]["claim"])

        # Re-lock
        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=hist)
        self.assertEqual(mgr.current_identity_id, "P1")
        self.assertEqual(self.gallery.list_identities()[0]["claim"], 1)

        # Test reset
        mgr.reset()
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertIsNone(self.gallery.list_identities()[0]["claim"])

    def test_09_person_id_tracker_adaptive_sync(self):
        """PersonIDTracker: maybe_adapt thanh cong va current_identity_id da gan -> goi gallery.add_template.
        Neu current_identity_id = None thi KHONG goi add_template."""
        from unittest.mock import patch, MagicMock
        import time

        settings.recognition["verify_interval_frames"] = 1
        settings.recognition["enroll_upgrade_enabled"] = False

        # Identity P1 co san 3 mau
        base_cluster = make_diverse_cluster(self.personA, n=3, noise=0.25)
        self.gallery.enroll("P1", base_cluster)
        self.assertEqual(self.gallery.get("P1")["n_templates"], 3)

        tracker = PersonIDTracker(detector=MagicMock(), recognizer=MagicMock(), identity_gallery=self.gallery)
        tracker.tracked_faces[1] = {
            "bbox": (10, 10, 50, 50),
            "gallery": MagicMock(),
            "lost": 0,
            "vx": 0.0,
            "vy": 0.0,
            "t": time.monotonic(),
        }
        tracker.tracked_faces[1]["gallery"].empty.return_value = False

        mock_prof = MagicMock()
        mock_prof.score.return_value = MagicMock(score=0.90)
        mock_prof.maybe_adapt.return_value = (True, "ok")
        tracker.target_profile = mock_prof

        # 1. current_identity_id = "P1" -> goi add_template thanh cong, templates tang len 4
        tracker.current_identity_id = "P1"
        new_feat = make_diverse_cluster(self.personA, n=1, noise=0.30)[0]
        with patch("person_id_tracker._safe_extract_feat", return_value=(new_feat, None, 0.9, None)):
            tracker.update([[10, 10, 50, 50]], np.zeros((100, 100, 3), dtype=np.uint8), locked_face_id=1)

        self.assertEqual(self.gallery.get("P1")["n_templates"], 4)

        # 2. current_identity_id = None -> KHONG goi add_template, templates giu nguyen 4
        tracker.current_identity_id = None
        new_feat2 = make_diverse_cluster(self.personA, n=1, noise=0.30)[0]
        with patch("person_id_tracker._safe_extract_feat", return_value=(new_feat2, None, 0.9, None)):
            tracker.update([[10, 10, 50, 50]], np.zeros((100, 100, 3), dtype=np.uint8), locked_face_id=1)

        self.assertEqual(self.gallery.get("P1")["n_templates"], 4)

    def test_10_ambiguous_identities_skip_merge_and_enroll(self):
        """Hai identity cung vuot nguong merge_sim (>= 0.50) nhung margin < 0.05:
        KHONG merge, KHÔNG enroll moi, log 'ambiguous', current_identity_id = None."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        # Tao hai vector p1 va p2 truc giao doi voi thanh phan nhieu nhung cung sim ~0.75 voi personA
        p = self.personA
        v1 = np.random.randn(512).astype(np.float32)
        v1 -= np.dot(v1, p) * p
        v1 /= np.linalg.norm(v1)

        v2 = np.random.randn(512).astype(np.float32)
        v2 -= np.dot(v2, p) * p
        v2 -= np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)

        p1 = 0.75 * p + 0.66 * v1; p1 /= np.linalg.norm(p1)
        p2 = 0.75 * p + 0.66 * v2; p2 /= np.linalg.norm(p2)

        # Enroll 2 identity vao gallery
        self.gallery.enroll("P1", [p1, p1, p1])
        self.gallery.enroll("P2", [p2, p2, p2])
        self.assertEqual(len(self.gallery.list_identities()), 2)

        # Lock muc tieu voi lich su la cac mau quanh personA
        hist_target = make_diverse_cluster(p, n=5, noise=0.10)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 1], 640, 480, history=hist_target, source="manual")

        # Ket qua phai la AMBIGUOUS: khong merge vao P1/P2, khong tao P3
        self.assertIsNone(mgr.current_identity_id)
        self.assertIsNone(face_tracker.current_identity_id)
        self.assertEqual(len(self.gallery.list_identities()), 2)
        self.assertEqual(self.gallery.get("P1")["n_templates"], 3)
        self.assertEqual(self.gallery.get("P2")["n_templates"], 3)

    def test_11_identity_logger_setup_and_handlers(self):
        """Kiem tra setup_identity_logging: co StreamHandler + FileHandler, propagate=False,
        khong bi gan trung handler khi goi lan 2, va ghi duoc log startup/messages ra file."""
        import tempfile
        import shutil
        from app import setup_identity_logging

        logger = logging.getLogger("identity")
        old_handlers = list(logger.handlers)
        for h in old_handlers:
            logger.removeHandler(h)

        temp_dir = tempfile.mkdtemp()
        try:
            log1 = setup_identity_logging(log_dir=temp_dir)
            self.assertEqual(log1.name, "identity")
            self.assertFalse(log1.propagate)
            self.assertEqual(len(log1.handlers), 2)
            has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log1.handlers)
            has_file = any(isinstance(h, logging.FileHandler) for h in log1.handlers)
            self.assertTrue(has_stream)
            self.assertTrue(has_file)

            # Goi lan 2 khong bi nhan doi handler
            log2 = setup_identity_logging(log_dir=temp_dir)
            self.assertEqual(len(log2.handlers), 2)

            # Ghi dong startup
            startup_msg = "[IDENTITY] enabled=True medoid_sim=0.40 merge_sim=0.50 min_fixed_sim=0.45"
            log1.info(startup_msg)

            # Flush handlers
            for h in log1.handlers:
                h.flush()

            log_file = os.path.join(temp_dir, "identity.log")
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(startup_msg, content)
        finally:
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)
            for h in old_handlers:
                logger.addHandler(h)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_12_lock_target_and_decisions_logged(self):
        """Kiem tra lock_target_by_bbox phat ra [IDENTITY] lock track=... va cac dong decision [IDENTITY]."""
        face_tracker = PersonIDTracker(recognizer=self.recognizer, identity_gallery=self.gallery)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker, identity_gallery=self.gallery)

        log_records = []
        class TestHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record.getMessage())

        th = TestHandler()
        id_logger = logging.getLogger("identity")
        id_logger.addHandler(th)
        try:
            # 1. Manual lock voi cluster hop le -> phat ra lock track=42 source=manual va decision='enroll'
            cluster = make_diverse_cluster(self.personA, n=5, noise=0.35)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 42], 640, 480, history=cluster, source="manual")

            lock_lines = [r for r in log_records if "[IDENTITY] lock track=42 source=manual" in r]
            self.assertTrue(len(lock_lines) >= 1)
            enroll_lines = [r for r in log_records if "[IDENTITY] decision='enroll'" in r]
            self.assertTrue(len(enroll_lines) >= 1)

            # 2. Auto lock -> phat ra lock track=99 source=auto va decision='skip', reason='source_not_manual (source=auto)'
            log_records.clear()
            mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300, 99], 640, 480, history=cluster, source="auto")
            auto_lock_lines = [r for r in log_records if "[IDENTITY] lock track=99 source=auto" in r]
            self.assertTrue(len(auto_lock_lines) >= 1)
            auto_skip_lines = [r for r in log_records if "[IDENTITY] decision='skip', reason='source_not_manual (source=auto)'" in r]
            self.assertTrue(len(auto_skip_lines) >= 1)

            # 3. Lock voi bbox khong co track id (len < 5) -> track=None
            log_records.clear()
            mgr.lock_target_by_bbox(dummy_frame, [100, 100, 200, 300], 640, 480, history=cluster, source="manual")
            no_id_lock_lines = [r for r in log_records if "[IDENTITY] lock track=None source=manual" in r]
            self.assertTrue(len(no_id_lock_lines) >= 1)
        finally:
            id_logger.removeHandler(th)


if __name__ == "__main__":
    unittest.main()

