#!/usr/bin/env python3
"""
test_profile_basic.py - Unit tests for TargetProfile and ArcFaceRecognizer enhancements
======================================================================================
Tests the 4 points:
1. TargetProfile.add_confuser: drops NaN/Inf/zero, rejects vector if fixed_score >= target_threshold.
2. ArcFaceRecognizer: self._lock around session.run, self.last_norm=0.0 in __init__, feature(img, return_norm).
3. TargetProfile.maybe_adapt: fixed_score >= add_sim (not score with bonus), margin >= 0.05.
4. _normalize & add templates: drops non-finite/zero-norm, infers dimension without hard-coded 512.
"""

import sys
import os
import threading
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

from target_profile import ProfileConfig, ProfileBuilder, TargetProfile, _normalize
from arcface_recognizer import ArcFaceRecognizer


def random_unit_vector(dim=512):
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Point 1: TargetProfile.add_confuser
# ---------------------------------------------------------------------------
def test_01_add_confuser_filters_invalid_and_high_score():
    cfg = ProfileConfig(add_sim=0.55, add_margin=0.10)
    target_vec = random_unit_vector(512)
    profile = TargetProfile(cfg, model_id="test_model", fixed_templates=[target_vec])

    # 1. Reject NaN vector
    nan_vec = np.full(512, np.nan, dtype=np.float32)
    profile.add_confuser(nan_vec)
    assert profile.stats()["confusers"] == 0

    # 2. Reject Inf vector
    inf_vec = np.full(512, np.inf, dtype=np.float32)
    profile.add_confuser(inf_vec)
    assert profile.stats()["confusers"] == 0

    # 3. Reject Zero vector
    zero_vec = np.zeros(512, dtype=np.float32)
    profile.add_confuser(zero_vec)
    assert profile.stats()["confusers"] == 0

    # 4. Reject vector with fixed_score >= target_threshold (the target itself)
    profile.add_confuser(target_vec, target_threshold=0.38)
    assert profile.stats()["confusers"] == 0

    # Similar vector to target (similarity ~ 0.90 >= 0.38)
    near_target = target_vec + 0.1 * random_unit_vector(512)
    near_target /= np.linalg.norm(near_target)
    profile.add_confuser(near_target, target_threshold=0.38)
    assert profile.stats()["confusers"] == 0

    # 5. Accept true confuser: orthogonal/dissimilar vector (similarity ~ 0.0 < 0.38)
    diff_vec = random_unit_vector(512)
    # Ensure low similarity with target
    while np.dot(target_vec, diff_vec) > 0.20:
        diff_vec = random_unit_vector(512)

    profile.add_confuser(diff_vec, target_threshold=0.38)
    assert profile.stats()["confusers"] == 1


# ---------------------------------------------------------------------------
# Point 2: ArcFaceRecognizer (_lock, last_norm, feature return_norm)
# ---------------------------------------------------------------------------
def test_02_arcface_recognizer_lock_and_return_norm():
    with patch("os.path.exists", return_value=True), \
         patch("onnxruntime.InferenceSession") as mock_session_cls:

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        # Return unnormalized raw vector of known norm (e.g. vector of 2s, norm = sqrt(512*4) = 45.2548)
        raw_feat = np.ones((1, 512), dtype=np.float32) * 2.0
        expected_norm = float(np.linalg.norm(raw_feat))
        mock_session.run.return_value = [raw_feat]
        mock_session_cls.return_value = mock_session

        rec = ArcFaceRecognizer("dummy_model.onnx", use_dml=False)

        # 1. Verify self._lock exists and is a Lock
        assert hasattr(rec, "_lock")
        assert isinstance(rec._lock, type(threading.Lock()))

        # 2. Verify self.last_norm is initialized to 0.0
        assert rec.last_norm == 0.0

        # Dummy aligned image (112x112x3 uint8)
        dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)

        # 3. Test feature(img, return_norm=False) -> returns feat only (shape (1, 512))
        feat = rec.feature(dummy_img, return_norm=False)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (1, 512)
        assert np.isclose(np.linalg.norm(feat), 1.0, atol=1e-5)
        assert np.isclose(rec.last_norm, expected_norm, atol=1e-4)

        # 4. Test feature(img, return_norm=True) -> returns (feat, norm)
        feat2, norm_val = rec.feature(dummy_img, return_norm=True)
        assert isinstance(feat2, np.ndarray)
        assert feat2.shape == (1, 512)
        assert isinstance(norm_val, float)
        assert np.isclose(norm_val, expected_norm, atol=1e-4)


# ---------------------------------------------------------------------------
# Point 3: TargetProfile.maybe_adapt (fixed_score >= add_sim, margin >= 0.05)
# ---------------------------------------------------------------------------
def test_03_maybe_adapt_fixed_score_and_margin():
    cfg = ProfileConfig(
        add_sim=0.60,
        add_margin=0.10,
        adapt_bonus=0.08,
        max_adaptive=5
    )
    target_base = random_unit_vector(512)
    profile = TargetProfile(cfg, model_id="test_model", fixed_templates=[target_base])

    # Case A: fixed_score is low (e.g. 0.55 < add_sim 0.60), but adaptive template could boost score
    # Manually seed an adaptive template that is close to candidate V
    cand_v = random_unit_vector(512)
    # Construct cand_v so that cosine(target_base, cand_v) = 0.55 (< add_sim 0.60)
    ortho_dir = cand_v - np.dot(cand_v, target_base) * target_base
    ortho_dir /= np.linalg.norm(ortho_dir)
    sim_target = 0.55
    cand_v = sim_target * target_base + np.sqrt(1.0 - sim_target**2) * ortho_dir
    cand_v /= np.linalg.norm(cand_v)

    # Put a template in adaptive that matches cand_v with 0.95
    # This gives adaptive_score = 0.95, bonus = min(0.08, 0.95 - 0.55) = 0.08
    # Total score = 0.55 + 0.08 = 0.63 >= 0.60.
    # BUT fixed_score is only 0.55 < 0.60!
    profile.adaptive_templates.append(cand_v.copy())
    profile.adaptive_images.append(None)

    # Now verify that maybe_adapt rejects because fixed_score < 0.60!
    added, reason = profile.maybe_adapt(cand_v)
    assert not added
    assert "score_low" in reason

    # Case B: fixed_score >= add_sim (0.75 >= 0.60), but margin < 0.05 due to confuser
    profile2 = TargetProfile(cfg, model_id="test_model", fixed_templates=[target_base])
    good_cand = 0.75 * target_base + np.sqrt(1.0 - 0.75**2) * ortho_dir
    good_cand /= np.linalg.norm(good_cand)

    # Add confuser very close to good_cand (e.g. sim with good_cand = 0.73)
    # s.fixed_score = 0.75, confuser_sim = 0.73 -> margin = 0.02 < 0.05
    profile2.confuser_templates.append(good_cand.copy())
    added2, reason2 = profile2.maybe_adapt(good_cand)
    assert not added2
    assert "margin_low" in reason2

    # Case C: fixed_score >= add_sim and margin >= 0.05 -> SUCCESS
    profile3 = TargetProfile(cfg, model_id="test_model", fixed_templates=[target_base])
    added3, reason3 = profile3.maybe_adapt(good_cand)
    assert added3
    assert reason3 == "added"


# ---------------------------------------------------------------------------
# Point 4: _normalize and template dim checking
# ---------------------------------------------------------------------------
def test_04_normalize_and_dim_inference():
    # 1. _normalize edge cases
    assert _normalize(None) is None
    assert _normalize([np.nan, 1.0, 2.0]) is None
    assert _normalize([np.inf, 1.0, 2.0]) is None
    assert _normalize(np.zeros(512)) is None
    assert _normalize(np.ones(512) * 1e-8) is None

    valid = _normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert valid is not None
    assert np.isclose(np.linalg.norm(valid), 1.0)
    assert np.allclose(valid, [0.6, 0.8], atol=1e-5)

    # 2. Dim inference without hardcoded 512: test with dim = 128
    dim128 = 128
    v128 = random_unit_vector(dim128)
    cfg = ProfileConfig(min_templates=2, target_templates=3)
    profile128 = TargetProfile(cfg, model_id="dim128_model", fixed_templates=[v128])
    assert profile128.dim == 128
    assert profile128.fixed_templates.shape == (1, 128)

    # 3. Dim mismatch handling
    v256 = random_unit_vector(256)

    # score returns 0.0 for mismatched dim
    s = profile128.score(v256)
    assert s.score == 0.0 and s.fixed_score == 0.0

    # maybe_adapt rejects mismatched dim
    ok_adapt, reason_adapt = profile128.maybe_adapt(v256)
    assert not ok_adapt
    assert reason_adapt == "dim_mismatch"

    # add_confuser rejects mismatched dim (confuser not added)
    profile128.add_confuser(v256)
    assert profile128.stats()["confusers"] == 0

    # ProfileBuilder dim tracking
    builder = ProfileBuilder(cfg, model_id="dim128_model")
    builder.add(v128)
    assert builder.dim == 128
    assert len(builder.samples) == 1

    # Adding mismatched vector to builder should be ignored
    builder.add(v256)
    assert len(builder.samples) == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
