#!/usr/bin/env python3
"""
test_profile_tracking_e2e.py - End-to-end unit test suite for TargetProfile integration
Tests TrackingManager + PersonIDTracker with profile_enabled = True.
"""

import sys
import os
import unittest
import time
import numpy as np

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from tracking import TrackingManager
from app import PersonIDTracker
from target_profile import ProfileConfig, TargetProfile


class MockRecognizer:
    """Mock recognizer supporting match() and feature() with pre-calculated vectors."""
    def match(self, f1, f2, mode=0):
        v1 = np.asarray(f1, dtype=np.float32).flatten()
        v2 = np.asarray(f2, dtype=np.float32).flatten()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))


def make_embedding(base_vec, noise=0.0):
    v = np.array(base_vec, dtype=np.float32)
    v /= np.linalg.norm(v)
    if noise > 0:
        noise_dir = np.random.randn(*v.shape).astype(np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        v = (1.0 - noise) * v + noise * noise_dir
        v /= np.linalg.norm(v)
    return v


class TestProfileTrackingE2E(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.personA = np.random.randn(512).astype(np.float32)
        self.personA /= np.linalg.norm(self.personA)

        self.personB = np.random.randn(512).astype(np.float32)
        self.personB /= np.linalg.norm(self.personB)

        self.recognizer = MockRecognizer()

    def test_01_lock_with_history_builds_ready_profile(self):
        """Kịch bản: Lock mục tiêu với lịch sử 6 mẫu của người A -> Profile đạt status 'ready'."""
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_min_templates"] = 3
        settings.recognition["profile_target_templates"] = 5
        settings.recognition["face_threshold"] = 0.38

        face_tracker = PersonIDTracker(recognizer=self.recognizer)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker)

        # Lịch sử 5 mẫu của người A
        history = [make_embedding(self.personA, noise=0.1) for _ in range(5)]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history)

        self.assertEqual(mgr.state, "TRACKING")
        self.assertEqual(mgr.locked_face_id, 1)
        self.assertEqual(mgr.profile_status, "ready")
        self.assertIsNotNone(mgr.profile)
        self.assertIsNotNone(face_tracker.target_profile)
        self.assertEqual(face_tracker.target_profile, mgr.profile)
        self.assertGreaterEqual(mgr.profile.stats()["fixed"], 3)

    def test_02_lock_without_history_learns_online(self):
        """Kịch bản: Lock khi chưa có lịch sử -> status 'learning' -> nạp đủ frame -> 'ready'."""
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_min_templates"] = 3
        settings.recognition["profile_target_templates"] = 4
        settings.recognition["face_threshold"] = 0.38

        face_tracker = PersonIDTracker(recognizer=self.recognizer)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        # Khóa không có history
        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=[])
        self.assertEqual(mgr.profile_status, "learning")
        self.assertIsNone(mgr.profile)

        # Trực tiếp thêm các mẫu vào builder (giả lập camera loop)
        for _ in range(4):
            feat = make_embedding(self.personA, noise=0.1)
            mgr.builder.add(feat)

        self.assertTrue(mgr.builder.ready())
        prof, info = mgr.builder.build()
        self.assertEqual(info["reason"], "ok")
        mgr.profile = prof
        mgr.profile_status = "ready"
        face_tracker.target_profile = prof

        self.assertEqual(mgr.profile_status, "ready")
        self.assertIsNotNone(mgr.profile)
        self.assertGreaterEqual(mgr.profile.stats()["fixed"], 3)

    def test_03_profile_preserves_on_lost_relocks_on_return(self):
        """Kịch bản: Người A ra khỏi khung > 25 frame (vào LOST), sau đó quay lại -> Profile được bảo toàn."""
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_target_templates"] = 4

        face_tracker = PersonIDTracker(recognizer=self.recognizer)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker)

        history = [make_embedding(self.personA, noise=0.1) for _ in range(4)]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history)
        self.assertEqual(mgr.profile_status, "ready")
        saved_profile = mgr.profile

        # Mô phỏng mất dấu vượt Grace Period (> 25 frames)
        for _ in range(30):
            res = mgr.update(dummy_frame, [], (480, 640))

        self.assertEqual(res["state"], "LOST")
        self.assertEqual(mgr.state, "LOST")
        # Profile PHẢI ĐƯỢC GIỮ NGUYÊN để relock
        self.assertIsNotNone(mgr.profile)
        self.assertEqual(mgr.profile, saved_profile)
        self.assertEqual(mgr.profile_status, "ready")

        # Người A quay lại với cùng ID = 1
        dets = [[110, 110, 210, 310, 1]]
        res2 = mgr.update(dummy_frame, dets, (480, 640))
        self.assertEqual(res2["state"], "TRACKING")
        self.assertEqual(mgr.locked_face_id, 1)
        # Profile vẫn là profile ban đầu
        self.assertEqual(mgr.profile, saved_profile)

    def test_04_mixed_history_falls_back_cleanly(self):
        """Kịch bản: Lịch sử bị lẫn 2 người khác nhau -> Builder phát hiện 'mixed' -> không crash, fallback an toàn."""
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_min_templates"] = 3
        settings.recognition["profile_target_templates"] = 6
        settings.recognition["profile_link_sim"] = 0.60
        settings.recognition["profile_learn_timeout_sec"] = 0.1

        face_tracker = PersonIDTracker(recognizer=self.recognizer)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker)

        # Lịch sử gồm 2 mẫu A và 2 mẫu B (bị lẫn do swap trước đó)
        mixed_history = [
            make_embedding(self.personA, noise=0.05),
            make_embedding(self.personA, noise=0.05),
            make_embedding(self.personB, noise=0.05),
            make_embedding(self.personB, noise=0.05),
        ]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=mixed_history)
        # Vì bị lẫn, không tạo được profile ngay
        self.assertEqual(mgr.profile_status, "learning")
        self.assertIsNone(mgr.profile)

        # Chờ quá timeout
        time.sleep(0.15)
        prof, info = mgr.builder.build()
        self.assertIn(info["reason"], ["mixed", "too_few_consistent"])
        mgr.profile_status = f"failed:{info['reason']}"

        self.assertTrue(mgr.profile_status.startswith("failed:"))
        self.assertIsNone(mgr.profile)
        # Vẫn chạy bình thường theo gallery cũ, không crash
        self.assertEqual(mgr.state, "TRACKING")

    def test_05_drop_target_and_reset_clears_profile(self):
        """Kịch bản: Drop target hoặc Reset -> xóa sạch profile và target_profile trong face_tracker."""
        settings.recognition["profile_enabled"] = True
        settings.recognition["profile_target_templates"] = 4
        face_tracker = PersonIDTracker(recognizer=self.recognizer)
        mgr = TrackingManager(None, self.recognizer, face_tracker=face_tracker)

        history = [make_embedding(self.personA, noise=0.1) for _ in range(4)]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo_bbox = [100, 100, 200, 300, 1]

        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history)
        self.assertIsNotNone(mgr.profile)
        self.assertIsNotNone(face_tracker.target_profile)

        # Test _drop_target
        mgr._drop_target("Test drop")
        self.assertIsNone(mgr.profile)
        self.assertIsNone(mgr.builder)
        self.assertEqual(mgr.profile_status, "none")
        self.assertIsNone(face_tracker.target_profile)

        # Test reset
        mgr.lock_target_by_bbox(dummy_frame, yolo_bbox, 640, 480, history=history)
        self.assertIsNotNone(mgr.profile)
        mgr.reset()
        self.assertIsNone(mgr.profile)
        self.assertIsNone(mgr.builder)
        self.assertEqual(mgr.profile_status, "none")
        self.assertIsNone(face_tracker.target_profile)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestProfileTrackingE2E)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nTất cả test E2E TargetProfile qua thành công!")
    else:
        sys.exit(1)
