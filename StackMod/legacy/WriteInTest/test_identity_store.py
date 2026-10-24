#!/usr/bin/env python3
"""
test_identity_store.py - Unit tests for IdentityStore
=====================================================
Tests:
1. save -> load match (meta + templates).
2. atomic save does not leave .tmp files.
3. wrong model_id or dim is ignored with log warning.
4. corrupt file does not crash.
5. delete works correctly.
6. invalid identity_id rejected by regex ^[A-Za-z0-9_-]{1,64}$ (raises ValueError).
7. error between 2 replaces preserves previously saved data (no corruption).
8. n_templates mismatch in meta.json is ignored.
9. save with no valid vectors does not write and returns False.
10. _replace_with_retry handles PermissionError on Windows with 3 retries.
11. delete path traversal protection.
"""

import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

import identity_store
from identity_store import IdentityStore, _replace_with_retry


def random_unit_vectors(n=4, dim=512):
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Test 1: save -> load match
# ---------------------------------------------------------------------------
def test_01_save_load_match(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)

    templates = random_unit_vectors(n=3, dim=512)
    ok = store.save(id="target_01", label="Person Alpha", templates=templates)
    assert ok is True

    # List ids
    ids = store.list_ids()
    assert ids == ["target_01"]

    # Load back
    loaded = store.load("target_01")
    assert loaded is not None
    meta, loaded_templates = loaded

    # Verify meta fields
    assert meta["id"] == "target_01"
    assert meta["label"] == "Person Alpha"
    assert meta["model_id"] == "arcface_w600k_r50"
    assert meta["dim"] == 512
    assert meta["n_templates"] == 3
    assert meta["version"] == 1
    assert "created_at" in meta

    # Verify templates
    assert loaded_templates.shape == (3, 512)
    assert loaded_templates.dtype == np.float32
    assert np.allclose(loaded_templates, templates, atol=1e-5)

    # Verify each row is L2 normalized
    norms = np.linalg.norm(loaded_templates, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Atomic save leaves no .tmp files
# ---------------------------------------------------------------------------
def test_02_atomic_save_no_tmp_left(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates = random_unit_vectors(n=2, dim=512)

    ok = store.save(id="target_atomic", label="Target Atomic", templates=templates)
    assert ok is True

    id_dir = tmp_path / "identities" / "target_atomic"
    assert id_dir.exists()

    # Check for any .tmp or .bak files
    tmp_files = list(id_dir.glob("*.tmp"))
    bak_files = list(id_dir.glob("*.bak"))
    assert len(tmp_files) == 0
    assert len(bak_files) == 0

    # Ensure final files exist
    assert (id_dir / "meta.json").exists()
    assert (id_dir / "templates.npy").exists()


# ---------------------------------------------------------------------------
# Test 3: Wrong model_id or dim is ignored
# ---------------------------------------------------------------------------
def test_03_wrong_model_id_or_dim_ignored(tmp_path):
    store_arcface = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates = random_unit_vectors(n=2, dim=512)
    store_arcface.save(id="target_compat", label="Compat Test", templates=templates)

    # 1. Store with different model_id (e.g. sface)
    store_sface = IdentityStore(root_dir=tmp_path, model_id="sface", dim=512)
    assert store_sface.load("target_compat") is None
    assert "target_compat" not in store_sface.list_ids()

    # 2. Store with different dim (e.g. 256)
    store_dim256 = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=256)
    assert store_dim256.load("target_compat") is None
    assert "target_compat" not in store_dim256.list_ids()

    # 3. Original store loads fine
    assert store_arcface.load("target_compat") is not None


# ---------------------------------------------------------------------------
# Test 4: Corrupt files do not crash
# ---------------------------------------------------------------------------
def test_04_corrupt_files_no_crash(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)

    # 1. Corrupt meta.json (invalid JSON content)
    corrupt_meta_dir = tmp_path / "identities" / "corrupt_meta"
    corrupt_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(corrupt_meta_dir / "meta.json", "w") as f:
        f.write("{invalid_json: true, broken")
    np.save(corrupt_meta_dir / "templates.npy", random_unit_vectors(1, 512))

    assert store.load("corrupt_meta") is None

    # 2. Corrupt templates.npy (truncated / invalid numpy file)
    corrupt_npy_dir = tmp_path / "identities" / "corrupt_npy"
    corrupt_npy_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "corrupt_npy",
        "label": "Corrupt NPY",
        "model_id": "arcface_w600k_r50",
        "dim": 512,
        "n_templates": 1,
    }
    with open(corrupt_npy_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    with open(corrupt_npy_dir / "templates.npy", "wb") as f:
        f.write(b"NOT_A_VALID_NPY_FILE_HEADER")

    assert store.load("corrupt_npy") is None


# ---------------------------------------------------------------------------
# Test 5: Delete works correctly
# ---------------------------------------------------------------------------
def test_05_delete_works(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates = random_unit_vectors(n=2, dim=512)

    store.save(id="target_del", label="To Delete", templates=templates)
    assert (tmp_path / "identities" / "target_del").exists()
    assert "target_del" in store.list_ids()

    # Delete existing
    ok = store.delete("target_del")
    assert ok is True
    assert not (tmp_path / "identities" / "target_del").exists()
    assert "target_del" not in store.list_ids()

    # Delete non-existing returns False
    assert store.delete("target_del") is False


# ---------------------------------------------------------------------------
# Test 6: Invalid identity_id rejected by regex (raises ValueError)
# ---------------------------------------------------------------------------
def test_06_invalid_id_rejected(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates = random_unit_vectors(n=1, dim=512)

    invalid_ids = [
        "../escape",
        "../../etc/passwd",
        "sub/folder",
        "has space",
        "has$char",
        "has#char",
        "",
        "a" * 65,  # > 64 chars
        None,
    ]

    for bad_id in invalid_ids:
        with pytest.raises(ValueError):
            store.save(bad_id, "label", templates)
        with pytest.raises(ValueError):
            store.load(bad_id)
        with pytest.raises(ValueError):
            store.delete(bad_id)

    # Valid IDs should work fine
    valid_ids = ["user_1", "User-99", "A", "123", "a" * 64]
    for vid in valid_ids:
        assert store._validate_id(vid) == vid


# ---------------------------------------------------------------------------
# Test 7: Error between 2 replaces preserves previously saved data
# ---------------------------------------------------------------------------
def test_07_error_between_replaces_preserves_previous_data(tmp_path, monkeypatch):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates_v1 = random_unit_vectors(n=2, dim=512)
    templates_v2 = random_unit_vectors(n=4, dim=512)

    # 1. Luu thanh cong ban ghi Version 1
    assert store.save("target_rollback", "Version 1", templates_v1) is True
    meta1, loaded1 = store.load("target_rollback")
    assert meta1["label"] == "Version 1"
    assert len(loaded1) == 2

    # 2. Intercept: buoc 1 (npy) thanh cong, buoc 2 (meta commit) bi loi PermissionError
    real_replace = identity_store._replace_with_retry

    def mock_replace(src, dst, retries=3, delay_sec=0.05):
        if "meta.json" in str(dst):
            raise PermissionError("Simulated failure during meta commit")
        return real_replace(src, dst, retries, delay_sec)

    monkeypatch.setattr(identity_store, "_replace_with_retry", mock_replace)

    # 3. Luu Version 2 bi fail
    ok = store.save("target_rollback", "Version 2", templates_v2)
    assert ok is False

    # 4. Kiem tra Version 1 khong he bi pha hong, van con nguyen ven
    monkeypatch.undo()
    meta_after, loaded_after = store.load("target_rollback")
    assert meta_after is not None
    assert meta_after["label"] == "Version 1"
    assert meta_after["n_templates"] == 2
    assert len(loaded_after) == 2
    assert np.allclose(loaded_after, templates_v1, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: n_templates mismatch in meta.json is ignored
# ---------------------------------------------------------------------------
def test_08_n_templates_mismatch_ignored(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates = random_unit_vectors(n=3, dim=512)
    assert store.save("target_mismatch", "Mismatch Test", templates) is True

    # Sua meta.json de n_templates lech voi thuc te (meta bao 5, thuc te chi co 3)
    meta_file = tmp_path / "identities" / "target_mismatch" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["n_templates"] = 5
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # load phai bo qua va tra ve None
    assert store.load("target_mismatch") is None
    assert "target_mismatch" not in store.list_ids()


# ---------------------------------------------------------------------------
# Test 9: Save with no valid vectors does not write and returns False
# ---------------------------------------------------------------------------
def test_09_save_no_valid_vectors_does_not_write(tmp_path):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)

    # Zero vectors
    zeros = np.zeros((3, 512), dtype=np.float32)
    assert store.save("target_zeros", "Zeros", zeros) is False
    assert not (tmp_path / "identities" / "target_zeros").exists()

    # NaN vectors
    nans = np.full((2, 512), np.nan, dtype=np.float32)
    assert store.save("target_nans", "NaNs", nans) is False
    assert not (tmp_path / "identities" / "target_nans").exists()

    # Empty array
    assert store.save("target_empty", "Empty", np.empty((0, 512), dtype=np.float32)) is False
    assert not (tmp_path / "identities" / "target_empty").exists()


# ---------------------------------------------------------------------------
# Test 10: _replace_with_retry handles PermissionError on Windows
# ---------------------------------------------------------------------------
def test_10_replace_retry_on_permission_error(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content", encoding="utf-8")

    attempts = 0
    real_replace = os.replace

    def mock_replace_eventually_succeeds(s, d):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("Locked by scanner on Windows")
        return real_replace(s, d)

    monkeypatch.setattr(os, "replace", mock_replace_eventually_succeeds)

    # Phai thanh cong o lan thu 3
    _replace_with_retry(src, dst, retries=3, delay_sec=0.01)
    assert attempts == 3
    assert dst.read_text(encoding="utf-8") == "content"

    # Neu that bai ca 3 lan thi phai raise PermissionError
    src2 = tmp_path / "src2.txt"
    dst2 = tmp_path / "dst2.txt"
    src2.write_text("content2", encoding="utf-8")
    attempts2 = 0

    def mock_replace_always_fails(s, d):
        nonlocal attempts2
        attempts2 += 1
        raise PermissionError("Always locked")

    monkeypatch.setattr(os, "replace", mock_replace_always_fails)
    with pytest.raises(PermissionError):
        _replace_with_retry(src2, dst2, retries=3, delay_sec=0.01)
    assert attempts2 == 3


# ---------------------------------------------------------------------------
# Test 11: Rollback loi van giu lai file .bak
# ---------------------------------------------------------------------------
def test_11_rollback_error_preserves_bak(tmp_path, monkeypatch):
    store = IdentityStore(root_dir=tmp_path, model_id="arcface_w600k_r50", dim=512)
    templates_v1 = random_unit_vectors(n=2, dim=512)
    templates_v2 = random_unit_vectors(n=3, dim=512)

    # 1. Luu thanh cong v1
    assert store.save("target_bak_test", "Version 1", templates_v1) is True

    # 2. Mock replace sao cho: commit meta loi, va rollback tu bak cung loi
    real_replace = identity_store._replace_with_retry

    def mock_replace(src, dst, retries=3, delay_sec=0.05):
        if "meta.json" in str(dst):
            raise PermissionError("Simulated error on meta commit")
        if "templates.npy.bak" in str(src):
            raise PermissionError("Simulated error on rollback from bak")
        return real_replace(src, dst, retries, delay_sec)

    monkeypatch.setattr(identity_store, "_replace_with_retry", mock_replace)

    # 3. Luu v2 that bai
    ok = store.save("target_bak_test", "Version 2", templates_v2)
    assert ok is False

    # 4. Kiem tra file templates.npy.bak van duoc giu lai tren dia
    bak_file = tmp_path / "identities" / "target_bak_test" / "templates.npy.bak"
    assert bak_file.exists()
