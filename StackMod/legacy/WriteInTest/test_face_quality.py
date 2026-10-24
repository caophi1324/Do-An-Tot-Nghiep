#!/usr/bin/env python3
"""
test_face_quality.py - Unit tests for Face Quality metrics and Best-of-N Anchor Upgrade
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

import face_utils
from face_utils import (
    cosine,
    pick_face,
    face_quality,
    blur_score,
    GalleryItem,
    FaceGallery,
)


class TestFaceQuality(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_01_cosine_pure_numpy(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        
        self.assertAlmostEqual(cosine(v1, v2), 0.0)
        self.assertAlmostEqual(cosine(v1, v3), 1.0)
        self.assertEqual(cosine(None, v1), -1.0)
        self.assertEqual(cosine(v1, None), -1.0)

    def test_02_pick_face_metrics(self):
        # YuNet format: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y, score]
        # Mặt chuẩn chính diện: w=60, h=60, re=(110, 110), le=(150, 110), nose=(130, 125), rm=(115, 140), lm=(145, 140), score=0.95
        # roi_w = 260 -> cx = 130
        good_face = np.array([100, 90, 60, 60, 110, 110, 150, 110, 130, 125, 115, 140, 145, 140, 0.95], dtype=np.float32)
        faces = [good_face]

        # 1. return_metrics=False (legacy)
        f_legacy = pick_face(faces, 260, return_metrics=False)
        self.assertIsNotNone(f_legacy)
        self.assertEqual(f_legacy[14], 0.95)

        # 2. return_metrics=True
        f, m = pick_face(faces, 260, return_metrics=True)
        self.assertIsNotNone(f)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["score"], 0.95)
        self.assertAlmostEqual(m["yaw"], 0.0)  # nose is centered between eyes
        # eye_mid_y = 110, mouth_mid_y = 140, nose_y = 125 -> pitch_ratio = 15/30 = 0.5
        self.assertAlmostEqual(m["pitch_ratio"], 0.5, places=2)

    def test_03_face_quality_computation(self):
        # Case 1: Mặt chất lượng cao (thẳng, nét, to, tỉ lệ chuẩn, norm cao)
        q_high = face_quality(score=0.98, w=100, h=100, yaw=0.02, pitch_ratio=0.55, blur=180.0, norm=24.0)
        self.assertGreater(q_high, 0.80)

        # Case 2: Mặt chất lượng thấp (mờ, nghiêng, nhỏ)
        q_low = face_quality(score=0.75, w=35, h=35, yaw=0.30, pitch_ratio=0.20, blur=45.0, norm=12.0)
        self.assertLess(q_low, 0.40)

        self.assertGreater(q_high, q_low)

    def test_04_gallery_item_quality(self):
        feat = np.ones(512, dtype=np.float32)
        item = GalleryItem(feat, quality=0.82)
        self.assertAlmostEqual(item.quality, 0.82)

    def test_05_gallery_try_upgrade_anchor(self):
        g = FaceGallery(maxlen=4)
        v1 = np.ones(512, dtype=np.float32)
        v1 /= np.linalg.norm(v1)

        # 1. Mẫu đầu tiên nạp vào làm anchor (chất lượng vừa 0.50)
        g.add(v1, quality=0.50, source="initial_enroll")
        self.assertEqual(len(g.items), 0)
        self.assertEqual(g.anchor_item.source, "initial_enroll")
        self.assertAlmostEqual(g.anchor_item.quality, 0.50)

        # 2. Mẫu thứ hai: người khác (cosine thấp ~0.1) -> từ chối nâng cấp
        v_other = np.zeros(512, dtype=np.float32)
        v_other[0] = 1.0
        upgraded = g.try_upgrade_anchor(v_other, quality=0.95, min_sim=0.55, min_gain=0.15)
        self.assertFalse(upgraded)
        self.assertAlmostEqual(g.anchor_item.quality, 0.50)

        # 3. Mẫu thứ ba: cùng người (cosine ~0.95) nhưng quality không vượt trội (0.55 < 0.50 + 0.15) -> từ chối nâng cấp
        v_same_low = v1.copy() + np.random.randn(512).astype(np.float32) * 0.05
        v_same_low /= np.linalg.norm(v_same_low)
        upgraded = g.try_upgrade_anchor(v_same_low, quality=0.55, min_sim=0.55, min_gain=0.15)
        self.assertFalse(upgraded)
        self.assertAlmostEqual(g.anchor_item.quality, 0.50)

        # 4. Mẫu thứ tư: cùng người (cosine ~0.95) và quality vượt trội (0.85 >= 0.50 + 0.15) -> CHẤP NHẬN nâng cấp!
        v_same_high = v1.copy() + np.random.randn(512).astype(np.float32) * 0.02
        v_same_high /= np.linalg.norm(v_same_high)
        upgraded = g.try_upgrade_anchor(v_same_high, quality=0.85, min_sim=0.55, min_gain=0.15)
        self.assertTrue(upgraded)
        self.assertAlmostEqual(g.anchor_item.quality, 0.85)
        self.assertEqual(g.anchor_item.source, "anchor_upgrade")

        # Anchor cũ PHẢI được hạ cấp xuống _items với source='demoted_anchor' chứ KHÔNG bị xoá mất
        self.assertEqual(len(g.items), 1)
        snapshot = g.snapshot()
        self.assertEqual(len(snapshot), 2)
        self.assertTrue(snapshot[0]["is_anchor"])
        self.assertEqual(snapshot[0]["source"], "anchor_upgrade")
        self.assertAlmostEqual(snapshot[0]["quality"], 0.85)

        self.assertFalse(snapshot[1]["is_anchor"])
        self.assertEqual(snapshot[1]["source"], "demoted_anchor")
        self.assertAlmostEqual(snapshot[1]["quality"], 0.50)


if __name__ == "__main__":
    unittest.main()
