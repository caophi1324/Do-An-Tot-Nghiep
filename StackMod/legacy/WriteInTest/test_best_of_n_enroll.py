#!/usr/bin/env python3
"""
test_best_of_n_enroll.py - Comprehensive Unit Tests for Best-of-N Face Enrollment
"""

import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

import face_utils
from face_utils import FaceGallery, GalleryItem, face_quality, cosine
from person_id_tracker import PersonIDTracker
from tracking import TrackingManager
from target_profile import ProfileConfig, ProfileBuilder
from config import settings


class MockDetector:
    def __init__(self, face_box=True):
        self.face_box = face_box

    def setInputSize(self, size):
        pass

    def detect(self, roi):
        if not self.face_box:
            return True, None
        h, w = roi.shape[:2]
        # YuNet format
        f = np.zeros((1, 15), dtype=np.float32)
        f[0, 0] = 5
        f[0, 1] = 5
        f[0, 2] = min(60, w - 10)
        f[0, 3] = min(60, h - 10)
        f[0, 4] = 20  # re_x
        f[0, 5] = 20  # re_y
        f[0, 6] = 40  # le_x
        f[0, 7] = 20  # le_y
        f[0, 8] = 30  # nose_x
        f[0, 9] = 30  # nose_y
        f[0, 10] = 22 # rm_x
        f[0, 11] = 40 # rm_y
        f[0, 12] = 38 # lm_x
        f[0, 13] = 40 # lm_y
        f[0, 14] = 0.95 # score
        return True, f


class MockRecognizer:
    def __init__(self):
        self.last_norm = 22.0

    def match(self, emb1, emb2, mode=0):
        return cosine(emb1, emb2)

    def alignCrop(self, frame, face):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        return np.ones(512, dtype=np.float32)


class TestBestOfNEnroll(unittest.TestCase):
    def setUp(self):
        settings.recognition["enroll_upgrade_enabled"] = True
        settings.recognition["enroll_window_sec"] = 2.0
        settings.recognition["enroll_max_candidates"] = 8
        settings.recognition["enroll_min_interval_sec"] = 0.05
        settings.recognition["anchor_upgrade_min_sim"] = 0.55
        settings.recognition["anchor_upgrade_min_gain"] = 0.15
        settings.recognition["enroll_max_extracts_per_frame"] = 1

    def test_01_two_phase_enroll_and_anchor_upgrade(self):
        """Kiểm tra quy trình enroll 2 pha: Anchor ban đầu được nâng cấp khi có mẫu tốt hơn."""
        tracker = PersonIDTracker(MockDetector(), MockRecognizer())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 200, 300]]  # Person 1

        v_base = np.zeros(512, dtype=np.float32)
        v_base[0] = 1.0

        current_face = [v_base.copy(), 0.40]  # (feat, quality)

        orig_extract = face_utils.extract_feat
        def fake_extract(*args, **kwargs):
            feat, q = current_face
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            metrics = {"score": 0.9, "w": 50, "h": 50, "yaw": 0.05, "pitch_ratio": 0.55, "blur": 100.0, "norm": 22.0}
            if kwargs.get("return_quality", False):
                return feat, img, q, metrics
            return feat, img

        face_utils.extract_feat = fake_extract
        try:
            # Frame 1: Tạo ID 1 với mẫu ban đầu (quality=0.40)
            res1 = tracker.update(boxes, frame)
            self.assertEqual(len(res1), 1)
            tid = res1[0][4]
            tr = tracker.tracked_faces[tid]
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.40)
            self.assertEqual(tr["gallery"].anchor_item.source, "enroll")
            self.assertEqual(len(tr["gallery"].items), 0)

            # Frame 2 (sau interval): Mẫu mới chất lượng cao hơn vượt bậc (quality=0.85, sim=1.0)
            time.sleep(0.06)
            current_face = [v_base.copy(), 0.85]
            tracker.update(boxes, frame)

            # Anchor PHẢI được nâng cấp lên quality=0.85
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.85)
            self.assertEqual(tr["gallery"].anchor_item.source, "anchor_upgrade")

            # Anchor cũ (quality=0.40) PHẢI được lưu lại thành demoted_anchor trong items
            self.assertEqual(len(tr["gallery"].items), 1)
            snapshot = tr["gallery"].snapshot()
            self.assertEqual(len(snapshot), 2)
            self.assertTrue(snapshot[0]["is_anchor"])
            self.assertEqual(snapshot[0]["quality"], 0.85)
            self.assertFalse(snapshot[1]["is_anchor"])
            self.assertEqual(snapshot[1]["source"], "demoted_anchor")
            self.assertEqual(snapshot[1]["quality"], 0.40)
        finally:
            face_utils.extract_feat = orig_extract

    def test_02_anchor_upgrade_rejected_on_low_sim_or_low_gain(self):
        """Từ chối nâng cấp anchor nếu sim < 0.55 (người khác) hoặc gain < 0.15 (không đủ tốt)."""
        tracker = PersonIDTracker(MockDetector(), MockRecognizer())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 200, 300]]

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0

        current_face = [v1.copy(), 0.60]

        orig_extract = face_utils.extract_feat
        def fake_extract(*args, **kwargs):
            feat, q = current_face
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            metrics = {"score": 0.9, "w": 50, "h": 50, "yaw": 0.05, "pitch_ratio": 0.55, "blur": 100.0, "norm": 22.0}
            if kwargs.get("return_quality", False):
                return feat, img, q, metrics
            return feat, img

        face_utils.extract_feat = fake_extract
        try:
            # Khởi tạo ID 1 với quality=0.60
            tracker.update(boxes, frame)
            tid = 1
            tr = tracker.tracked_faces[tid]
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.60)

            # Trường hợp 1: Người khác (vector vuông góc, sim=0.0) dù quality=0.99 vẫn BỊ TỪ CHỐI
            time.sleep(0.06)
            v_other = np.zeros(512, dtype=np.float32)
            v_other[1] = 1.0
            current_face = [v_other, 0.99]
            tracker.update(boxes, frame)
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.60)
            self.assertEqual(tr["gallery"].anchor_item.source, "enroll")

            # Trường hợp 2: Cùng người (sim=1.0) nhưng gain nhỏ (0.65 < 0.60 + 0.15) -> BỊ TỪ CHỐI
            time.sleep(0.06)
            current_face = [v1.copy(), 0.65]
            tracker.update(boxes, frame)
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.60)
        finally:
            face_utils.extract_feat = orig_extract

    def test_03_enroll_upgrade_disabled_flag(self):
        """Khi enroll_upgrade_enabled = False, hệ thống giữ nguyên anchor ban đầu và không nâng cấp."""
        settings.recognition["enroll_upgrade_enabled"] = False
        tracker = PersonIDTracker(MockDetector(), MockRecognizer())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 200, 300]]

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        current_face = [v1.copy(), 0.30]

        orig_extract = face_utils.extract_feat
        def fake_extract(*args, **kwargs):
            feat, q = current_face
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            metrics = {"score": 0.9, "w": 50, "h": 50, "yaw": 0.05, "pitch_ratio": 0.55, "blur": 100.0, "norm": 22.0}
            if kwargs.get("return_quality", False):
                return feat, img, q, metrics
            return feat, img

        face_utils.extract_feat = fake_extract
        try:
            tracker.update(boxes, frame)
            tr = tracker.tracked_faces[1]
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.30)

            # Đưa mẫu hoàn hảo vào
            time.sleep(0.06)
            current_face = [v1.copy(), 0.95]
            tracker.update(boxes, frame)

            # Vì enroll_upgrade_enabled = False, anchor không bị đổi qua cơ chế Step 3.5
            self.assertEqual(tr["gallery"].anchor_item.quality, 0.30)
            self.assertEqual(tr["gallery"].anchor_item.source, "enroll")
        finally:
            face_utils.extract_feat = orig_extract
            settings.recognition["enroll_upgrade_enabled"] = True

    def test_04_lock_target_inherits_gallery_history(self):
        """Khóa mục tiêu kế thừa toàn bộ gallery và ảnh từ history thay vì xóa sạch."""
        tm = TrackingManager(MockDetector(), MockRecognizer())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = [100, 100, 200, 300, 1]

        v1 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[1] = 1.0
        dummy_img = np.ones((96, 96, 3), dtype=np.uint8)

        # History gồm anchor cũ và 1 item
        history = [
            (v1, dummy_img, 1.0, "enroll", time.time()),
            (v2, dummy_img, 0.8, "learn", time.time()),
        ]

        # Giả lập frame hiện tại không thấy rõ mặt (feat=None)
        orig_extract = face_utils.extract_feat
        def fake_extract(*args, **kwargs):
            if kwargs.get("return_quality", False):
                return None, None, 0.0, None
            return None, None

        face_utils.extract_feat = fake_extract
        try:
            tm.lock_target_by_bbox(frame, box, 640, 480, history=history)

            # Gallery của tm PHẢI kế thừa 2 mẫu từ history
            self.assertFalse(tm.gallery.empty())
            self.assertTrue(np.allclose(tm.gallery.anchor, v1))
            self.assertEqual(len(tm.gallery.items), 1)
            self.assertTrue(np.allclose(tm.gallery.items[0], v2))
            self.assertEqual(tm.decision_source, "khuôn mặt")
            self.assertIsNotNone(tm.cached_face_img)
        finally:
            face_utils.extract_feat = orig_extract

    def test_05_profile_builder_medoid_and_diversity(self):
        """ProfileBuilder chọn top-N theo medoid và farthest-point sampling."""
        cfg = ProfileConfig(min_templates=3, target_templates=4, link_sim=0.4)
        builder = ProfileBuilder(cfg, model_id="arcface_test")

        np.random.seed(123)
        base = np.random.randn(512).astype(np.float32)
        base /= np.linalg.norm(base)

        for i in range(6):
            noise_dir = np.random.randn(512).astype(np.float32)
            noise_dir /= np.linalg.norm(noise_dir)
            s = 0.85 * base + 0.15 * noise_dir
            s /= np.linalg.norm(s)
            builder.add(s)

        self.assertTrue(builder.ready())
        profile, info = builder.build()
        self.assertIsNotNone(profile)
        self.assertEqual(info["reason"], "ok")
        # Giới hạn tối đa target_templates = 4
        self.assertEqual(len(profile.fixed_templates), 4)


if __name__ == "__main__":
    unittest.main()
