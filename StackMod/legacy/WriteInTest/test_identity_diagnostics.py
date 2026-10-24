#!/usr/bin/env python3
"""
test_identity_diagnostics.py - Test suite for Bước 5c.0 diagnostic logging:
1. [SWAP-DETAIL] and self.last_swap_event in PersonIDTracker._verify_and_swap.
2. [LOCK-SWAP] in TrackingManager.update() when swap involves locked_face_id.
3. [MISMATCH] in TrackingManager.update() when mismatch increments or resets.
4. [RELOCK-ID] in TrackingManager.update() (LOST state Priority 1) with extract_feat throttle.
"""

import sys
import os
import time
import threading
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import logging

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


class TestIdentityDiagnostics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dim = 512
        self.vecA = np.zeros(self.dim, dtype=np.float32)
        self.vecA[0] = 1.0

        self.vecB = np.zeros(self.dim, dtype=np.float32)
        self.vecB[1] = 1.0

        self.recognizer = MockRecognizer(dim=self.dim)
        self.detector = MockDetector()

        self.orig_recognition = dict(settings.recognition)
        self.orig_tracking = dict(settings.tracking)
        settings.recognition["identity_enabled"] = True

        # Mock event loggers so id_events.csv is not written
        self.patch_person_id_logger = patch.object(person_id_tracker.id_logger, 'log_event', return_value=None)
        self.patch_person_id_logger.start()
        self.patch_tracking_id_logger = patch.object(tracking.id_logger, 'log_event', return_value=None)
        self.patch_tracking_id_logger.start()

        # Capture "identity" logs
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
        settings.recognition.clear()
        settings.recognition.update(self.orig_recognition)
        settings.tracking.clear()
        settings.tracking.update(self.orig_tracking)

        self.patch_person_id_logger.stop()
        self.patch_tracking_id_logger.stop()

        self.id_logger.removeHandler(self.log_handler)
        for h in self.removed_handlers:
            self.id_logger.addHandler(h)
        self.id_logger.setLevel(self.orig_log_level)

    def test_swap_detail_and_last_swap_event(self):
        """Test [SWAP-DETAIL] log and self.last_swap_event in PersonIDTracker._verify_and_swap."""
        tracker = PersonIDTracker(detector=self.detector, recognizer=self.recognizer)
        
        # Setup track 1 with vecA in gallery, track 2 with vecB in gallery
        g1 = FaceGallery()
        g1.add(self.vecA)
        tracker.tracked_faces[1] = {"bbox": [10, 10, 50, 50], "lost": 0, "t": 0.0, "gallery": g1, "history": [self.vecA]}

        g2 = FaceGallery()
        g2.add(self.vecB)
        tracker.tracked_faces[2] = {"bbox": [100, 100, 140, 140], "lost": 0, "t": 0.0, "gallery": g2, "history": [self.vecB]}

        # Box 0 is assigned track 1, but its face matches track 2 (vecB)
        # Box 1 is assigned track 2, but its face matches track 1 (vecA)
        assign = {0: 1, 1: 2}
        boxes = [[10, 10, 50, 50], [100, 100, 140, 140]]
        crowded = [False, False]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        settings.recognition["verify_interval_frames"] = 1
        settings.recognition["swap_votes"] = 1

        def mock_extract(det, rec, fr, b, r, lk):
            # If box is around (10, 10), return vecB
            if b[0] < 50:
                return self.vecB, np.zeros((20, 20, 3), dtype=np.uint8), 1.0, 0.0
            else:
                return self.vecA, np.zeros((20, 20, 3), dtype=np.uint8), 1.0, 0.0

        with patch("person_id_tracker._safe_extract_feat", side_effect=mock_extract):
            swapped = tracker._verify_and_swap(assign, boxes, crowded, dummy_frame, None, 0.45, 0.363, locked_face_id=1)

        self.assertIn(1, swapped)
        self.assertIn(2, swapped)
        self.assertIsNotNone(tracker.last_swap_event)
        self.assertEqual(tracker.last_swap_event["cur"], 1)
        self.assertEqual(tracker.last_swap_event["best"], 2)

        # Check [SWAP-DETAIL] log
        swap_logs = [m for m in self.log_handler.records if "[SWAP-DETAIL]" in m]
        self.assertTrue(len(swap_logs) >= 1)
        self.assertIn("cur=1", swap_logs[0])
        self.assertIn("best=2", swap_logs[0])
        self.assertIn("locked=1", swap_logs[0])

    def test_lock_swap_in_tracking_manager(self):
        """Test [LOCK-SWAP] log in TrackingManager when swap involves locked_face_id."""
        rec_lock = threading.Lock()
        gallery = IdentityGallery(dim=self.dim)
        face_tracker = PersonIDTracker(detector=self.detector, recognizer=self.recognizer,
                                       identity_gallery=gallery)
        mgr = TrackingManager(detector=self.detector, recognizer=self.recognizer,
                              rec_lock=rec_lock, face_tracker=face_tracker, identity_gallery=gallery)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Lock target ID 1
        mgr.lock_target_by_bbox(frame, [100, 100, 200, 200, 1], 640, 480, source="manual")
        self.assertEqual(mgr.state, "TRACKING")
        self.assertEqual(mgr.locked_face_id, 1)

        # Simulate swap event on face_tracker involving ID 1
        face_tracker.last_swap_event = {
            "cur": 1,
            "best": 2,
            "time": time.monotonic()
        }

        # Next update with detection of ID 1 at a shifted position
        detections = [[250, 100, 350, 200, 1]]
        mgr.update(frame, detections, frame_shape=(480, 640))

        lock_swap_logs = [m for m in self.log_handler.records if "[LOCK-SWAP]" in m]
        self.assertEqual(len(lock_swap_logs), 1)
        self.assertIn("locked_id=1", lock_swap_logs[0])
        self.assertIn("dx=", lock_swap_logs[0])

    def test_mismatch_logging_increment_and_reset(self):
        """Test [MISMATCH] log when mismatch increments and resets."""
        rec_lock = threading.Lock()
        gallery = IdentityGallery(dim=self.dim)
        mgr = TrackingManager(detector=self.detector, recognizer=self.recognizer,
                              rec_lock=rec_lock, identity_gallery=gallery)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Lock target ID 1 and populate gallery with vecA
        mgr.lock_target_by_bbox(frame, [100, 100, 200, 200, 1], 640, 480, source="manual")
        mgr.gallery.add(self.vecA)
        mgr.locked_face_id = 1
        settings.recognition["recognition_interval_frames"] = 1
        settings.recognition["face_threshold"] = 0.50

        # Mock _safe_extract_feat to return vecB (similarity 0.0 with vecA) -> mismatch increments
        with patch("tracking._safe_extract_feat", return_value=(self.vecB, np.zeros((10, 10, 3)), 1.0, 0.0)):
            detections = [[100, 100, 200, 200, 1]]
            mgr.update(frame, detections, frame_shape=(480, 640))

        # Check MISMATCH log increment
        mismatch_inc_logs = [m for m in self.log_handler.records if "[MISMATCH] n=1" in m]
        self.assertEqual(len(mismatch_inc_logs), 1)
        self.assertIn("id=1", mismatch_inc_logs[0])

        # Mock _safe_extract_feat to return vecA (similarity 1.0 >= 0.50) -> resets mismatch
        with patch("tracking._safe_extract_feat", return_value=(self.vecA, np.zeros((10, 10, 3)), 1.0, 0.0)):
            mgr.update(frame, detections, frame_shape=(480, 640))

        mismatch_reset_logs = [m for m in self.log_handler.records if "[MISMATCH] n=0" in m]
        self.assertTrue(len(mismatch_reset_logs) >= 1)
        self.assertEqual(mgr.mismatch, 0)

    def test_relock_id_logging_in_lost_state(self):
        """Test [RELOCK-ID] log in LOST state priority 1 and face_score extraction."""
        rec_lock = threading.Lock()
        gallery = IdentityGallery(dim=self.dim)
        mgr = TrackingManager(detector=self.detector, recognizer=self.recognizer,
                              rec_lock=rec_lock, identity_gallery=gallery)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mgr.lock_target_by_bbox(frame, [100, 100, 200, 200, 1], 640, 480, source="manual")
        mgr.gallery.add(self.vecA)
        mgr._raw_locked_bbox = [100, 100, 200, 200]
        mgr.state = "LOST"
        mgr.locked_face_id = 1
        mgr.lost_frame_counter = 15

        # Detection with ID 1 reappears
        detections = [[120, 100, 220, 200, 1]]
        with patch("tracking._safe_extract_feat", return_value=(self.vecA, np.zeros((10, 10, 3)), 1.0, 0.0)) as mock_ext:
            mgr.update(frame, detections, frame_shape=(480, 640))
            self.assertEqual(mock_ext.call_count, 1)

        relock_logs = [m for m in self.log_handler.records if "[RELOCK-ID]" in m]
        self.assertEqual(len(relock_logs), 1)
        self.assertIn("track=1", relock_logs[0])
        self.assertIn("lost_frames=15", relock_logs[0])
        self.assertIn("crowded=False", relock_logs[0])
        self.assertIn("face_score=", relock_logs[0])
        self.assertEqual(mgr.state, "TRACKING")

    def test_reject_counts_all_reasons(self):
        """Test face_utils.REJECT_COUNTS increments for no_face, pick_face (score/small/yaw/offset), blur, and ok."""
        import face_utils
        face_utils.get_and_reset_extract_counts()

        class DummyDet:
            def __init__(self, faces):
                self.faces = faces
            def setInputSize(self, sz):
                pass
            def detect(self, roi):
                return True, self.faces

        class DummyRec:
            def alignCrop(self, roi, f):
                # Return noisy image so blur_score is high
                return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            def feature(self, img):
                return np.zeros(512, dtype=np.float32)

        class BlurryRec(DummyRec):
            def alignCrop(self, roi, f):
                # Solid black has 0.0 blur score
                return np.zeros((112, 112, 3), dtype=np.uint8)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [100, 100, 300, 400]
        # roi_w = 200, center = 100

        # 1. No face
        det_no_face = DummyDet([])
        face_utils.extract_feat(det_no_face, DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["no_face"], 1)

        # 2. Pick face: low score
        # [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y, score]
        low_score_face = np.array([75, 20, 50, 50, 85, 35, 115, 35, 100, 45, 90, 55, 110, 55, 0.50], dtype=np.float32)
        face_utils.extract_feat(DummyDet([low_score_face]), DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["pick_face"], 1)
        self.assertEqual(face_utils.REJECT_COUNTS["score"], 1)

        # 3. Pick face: small size
        small_face = np.array([75, 20, 25, 25, 80, 25, 95, 25, 87, 32, 82, 38, 93, 38, 0.95], dtype=np.float32)
        face_utils.extract_feat(DummyDet([small_face]), DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["pick_face"], 2)
        self.assertEqual(face_utils.REJECT_COUNTS["small"], 1)

        # 4. Pick face: high yaw (nose at 125 instead of center 100 between 85 and 115 -> dx=30, diff=25/30 = 0.833 > max_yaw)
        yaw_face = np.array([75, 20, 50, 50, 85, 35, 115, 35, 125, 45, 90, 55, 110, 55, 0.95], dtype=np.float32)
        face_utils.extract_feat(DummyDet([yaw_face]), DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["pick_face"], 3)
        self.assertEqual(face_utils.REJECT_COUNTS["yaw"], 1)

        # 5. Pick face: high offset (x=0, w=50 -> center=25, roi_w=200, roi_center=100 -> off = |25-100|/200 = 0.375 > 0.30)
        off_face = np.array([0, 20, 50, 50, 10, 35, 40, 35, 25, 45, 15, 55, 35, 55, 0.95], dtype=np.float32)
        face_utils.extract_feat(DummyDet([off_face]), DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["pick_face"], 4)
        self.assertEqual(face_utils.REJECT_COUNTS["offset"], 1)

        # 6. Blur fail: sharp=False
        good_face = np.array([75, 20, 50, 50, 85, 35, 115, 35, 100, 45, 90, 55, 110, 55, 0.95], dtype=np.float32)
        face_utils.extract_feat(DummyDet([good_face]), BlurryRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["blur"], 1)

        # 7. OK success
        face_utils.extract_feat(DummyDet([good_face]), DummyRec(), frame, bbox, 0.45)
        self.assertEqual(face_utils.REJECT_COUNTS["ok"], 1)

        # Verify get_and_reset_extract_counts
        counts = face_utils.get_and_reset_extract_counts()
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(counts["no_face"], 1)
        self.assertEqual(counts["pick_face"], 4)
        self.assertEqual(counts["blur"], 1)
        self.assertEqual(counts["total"], 7)
        self.assertEqual(counts["score"], 1)
        self.assertEqual(counts["small"], 1)
        self.assertEqual(counts["yaw"], 1)
        self.assertEqual(counts["offset"], 1)

        # After reset, all counts are 0
        for k, v in face_utils.REJECT_COUNTS.items():
            self.assertEqual(v, 0)

    def test_extract_rate_10s_logging(self):
        """Test PersonIDTracker logs [EXTRACT-RATE] every 10s and resets counters."""
        import face_utils
        face_utils.get_and_reset_extract_counts()
        face_utils.REJECT_COUNTS["ok"] = 10
        face_utils.REJECT_COUNTS["no_face"] = 5
        face_utils.REJECT_COUNTS["pick_face"] = 3
        face_utils.REJECT_COUNTS["blur"] = 2

        tracker = PersonIDTracker(detector=self.detector, recognizer=self.recognizer)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Under 10s: no log
        tracker._last_extract_rate_time = time.monotonic() - 5.0
        tracker.update([], dummy_frame)
        extract_logs = [m for m in self.log_handler.records if "[EXTRACT-RATE]" in m]
        self.assertEqual(len(extract_logs), 0)

        # Over 10s: logs and resets
        tracker._last_extract_rate_time = time.monotonic() - 11.0
        tracker.update([], dummy_frame)
        extract_logs = [m for m in self.log_handler.records if "[EXTRACT-RATE]" in m]
        self.assertEqual(len(extract_logs), 1)
        self.assertIn("ok=10", extract_logs[0])
        self.assertIn("no_face=5", extract_logs[0])
        self.assertIn("pick_face=3", extract_logs[0])
        self.assertIn("blur=2", extract_logs[0])
        self.assertIn("total=20", extract_logs[0])

        # Verify counters were reset
        self.assertEqual(face_utils.REJECT_COUNTS["ok"], 0)
        self.assertEqual(face_utils.REJECT_COUNTS["total"] if "total" in face_utils.REJECT_COUNTS else 0, 0)


if __name__ == "__main__":
    unittest.main()

