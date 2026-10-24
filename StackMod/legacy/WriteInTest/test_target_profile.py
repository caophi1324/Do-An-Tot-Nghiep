#!/usr/bin/env python3
"""
test_target_profile.py - Unit test suite for TargetProfile and ProfileBuilder
"""

import sys
import os
import unittest
import numpy as np

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from target_profile import ProfileConfig, ProfileBuilder, TargetProfile, Score


def make_embedding(base_vec, noise=0.0):
    v = np.array(base_vec, dtype=np.float32)
    v /= np.linalg.norm(v)
    if noise > 0:
        noise_dir = np.random.randn(*v.shape).astype(np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        v = (1.0 - noise) * v + noise * noise_dir
        v /= np.linalg.norm(v)
    return v


class TestTargetProfile(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.personA_base = np.random.randn(512).astype(np.float32)
        self.personA_base /= np.linalg.norm(self.personA_base)

        self.personB_base = np.random.randn(512).astype(np.float32)
        self.personB_base /= np.linalg.norm(self.personB_base)

    def test_01_config_from_dict(self):
        d = {
            "profile_min_templates": 4,
            "profile_target_templates": 8,
            "profile_link_sim": 0.45,
            "profile_dup_sim": 0.90,
            "profile_add_sim": 0.60,
            "profile_add_margin": 0.12,
            "profile_adapt_bonus": 0.05,
            "profile_max_adaptive": 10,
            "profile_top_k": 4,
            "profile_learn_timeout_sec": 7.5,
        }
        cfg = ProfileConfig.from_dict(d)
        self.assertEqual(cfg.min_templates, 4)
        self.assertEqual(cfg.target_templates, 8)
        self.assertAlmostEqual(cfg.link_sim, 0.45)
        self.assertAlmostEqual(cfg.dup_sim, 0.90)
        self.assertAlmostEqual(cfg.add_sim, 0.60)
        self.assertAlmostEqual(cfg.add_margin, 0.12)
        self.assertAlmostEqual(cfg.adapt_bonus, 0.05)
        self.assertEqual(cfg.max_adaptive, 10)
        self.assertEqual(cfg.top_k, 4)
        self.assertAlmostEqual(cfg.learn_timeout_sec, 7.5)

    def test_02_builder_ready_and_build_ok(self):
        cfg = ProfileConfig(min_templates=3, target_templates=4, link_sim=0.5)
        builder = ProfileBuilder(cfg, model_id="arcface_w600k_r50")

        self.assertFalse(builder.ready())
        self.assertEqual(builder.progress(), (0, 4))

        # Thêm 4 mẫu của cùng người A (noise nhỏ, sim > 0.8)
        samples = [make_embedding(self.personA_base, noise=0.15) for _ in range(4)]
        for s in samples:
            builder.add(s)

        self.assertTrue(builder.ready())
        self.assertEqual(builder.progress()[0], 4)

        profile, info = builder.build()
        self.assertIsNotNone(profile)
        self.assertEqual(info["reason"], "ok")
        self.assertGreaterEqual(info["n_kept"], 3)
        self.assertEqual(profile.stats()["fixed"], info["n_kept"])

    def test_03_builder_too_few(self):
        cfg = ProfileConfig(min_templates=3, target_templates=5)
        builder = ProfileBuilder(cfg, model_id="arcface")
        builder.add(make_embedding(self.personA_base))
        profile, info = builder.build()
        self.assertIsNone(profile)
        self.assertEqual(info["reason"], "too_few")

    def test_04_builder_detects_mixed_or_inconsistent(self):
        cfg = ProfileConfig(min_templates=3, target_templates=6, link_sim=0.6)
        builder = ProfileBuilder(cfg, model_id="arcface")

        # Đưa lẫn 2 mẫu người A và 2 mẫu người B (sim giữa A và B ~ 0)
        builder.add(make_embedding(self.personA_base, noise=0.05))
        builder.add(make_embedding(self.personA_base, noise=0.05))
        builder.add(make_embedding(self.personB_base, noise=0.05))
        builder.add(make_embedding(self.personB_base, noise=0.05))

        profile, info = builder.build()
        self.assertIn(info["reason"], ["mixed", "too_few_consistent"])

    def test_05_profile_score_and_check(self):
        cfg = ProfileConfig(top_k=2)
        templates = [make_embedding(self.personA_base, noise=0.1) for _ in range(3)]
        profile = TargetProfile(cfg, model_id="arcface", fixed_templates=templates)

        # Chấm điểm với mẫu mới của người A
        testA = make_embedding(self.personA_base, noise=0.1)
        sA = profile.score(testA)
        self.assertIsInstance(sA, Score)
        self.assertGreater(sA.score, 0.70)
        self.assertGreater(sA.fixed_score, 0.70)

        ok, _ = profile.check(testA, threshold=0.50, margin_min=0.05)
        self.assertTrue(ok)

        # Chấm điểm với người B
        testB = make_embedding(self.personB_base, noise=0.1)
        sB = profile.score(testB)
        self.assertLess(sB.score, 0.35)

        ok_B, _ = profile.check(testB, threshold=0.50, margin_min=0.05)
        self.assertFalse(ok_B)

    def test_06_profile_adaptation_and_confuser(self):
        cfg = ProfileConfig(add_sim=0.60, add_margin=0.10, adapt_bonus=0.04, max_adaptive=3)
        templates = [make_embedding(self.personA_base, noise=0.1) for _ in range(3)]
        profile = TargetProfile(cfg, model_id="arcface", fixed_templates=templates)

        self.assertEqual(profile.stats()["adaptive"], 0)

        # 1. Thêm mẫu tốt cùng người A (không có người khác đứng gần)
        good_A = make_embedding(self.personA_base, noise=0.2)
        added, why = profile.maybe_adapt(good_A, others_max_sim=0.20)
        self.assertTrue(added)
        self.assertEqual(why, "added")
        self.assertEqual(profile.stats()["adaptive"], 1)

        # 2. Thử thêm mẫu bị người khác tranh chấp (margin không đủ)
        contested_A = make_embedding(self.personA_base, noise=0.25)
        added2, why2 = profile.maybe_adapt(contested_A, others_max_sim=0.90)
        self.assertFalse(added2)
        self.assertIn("margin_insufficient", why2)

        # 3. Thêm confuser (người B đứng gần)
        profile.add_confuser(self.personB_base, target_threshold=0.38)
        self.assertGreaterEqual(profile.stats()["confusers"], 1)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTargetProfile)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nTất cả test qua")
    else:
        sys.exit(1)
