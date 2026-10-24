#!/usr/bin/env python3
"""
test_identity_gallery.py - Unit tests for IdentityGallery
=========================================================
Tests:
1. hai identity gia lap khac nhau match dung.
2. margin thap khi hai identity gan nhau.
3. select_diverse khong chon mau gan trung.
4. claim doc quyen, release roi claim lai duoc.
5. add_template tu choi mau trung va dung o 12 ma fixed con nguyen.
6. feature NaN/Inf/zero/sai dim khong crash.
7. clear("manual_lock") giu lai identity "disk".
8. remove va release_track don sach claims dung quy dinh.
9. da luong (concurrent access) khong xay ra deadlock hoac xung dot du lieu.
"""

import os
import sys
import threading
import numpy as np
import pytest

# Ensure pc_client is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

from identity_gallery import IdentityGallery


def make_unit_vector(dim=512, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float32)
    else:
        v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Test 1: Hai identity gia lap khac nhau match dung
# ---------------------------------------------------------------------------
def test_01_two_different_identities_match_correctly():
    gallery = IdentityGallery(min_gap_sim=0.92, max_templates=12, top_k=3, dim=512)

    # Tao 2 vector co so truc giao nhau (cosine sim = 0.0)
    v_a = np.zeros(512, dtype=np.float32)
    v_a[0] = 1.0
    v_b = np.zeros(512, dtype=np.float32)
    v_b[1] = 1.0

    gallery.enroll("id_A", [v_a], label="Person A", source="manual_lock")
    gallery.enroll("id_B", [v_b], label="Person B", source="manual_lock")

    # Vector truy van gan id_A (them nhieu nhe)
    query_a = v_a.copy()
    query_a[2] = 0.05
    query_a /= np.linalg.norm(query_a)

    best_id, score, margin = gallery.match(query_a)
    assert best_id == "id_A"
    assert score > 0.99
    # Vi B truc giao voi A (sim ~ 0), margin = score_A - score_B ~ 0.99
    assert margin > 0.9

    # Vector truy van gan id_B
    query_b = v_b.copy()
    query_b[2] = 0.05
    query_b /= np.linalg.norm(query_b)

    best_id, score, margin = gallery.match(query_b)
    assert best_id == "id_B"
    assert score > 0.99
    assert margin > 0.9


# ---------------------------------------------------------------------------
# Test 2: Margin thap khi hai identity gan nhau
# ---------------------------------------------------------------------------
def test_02_margin_low_when_identities_are_close():
    gallery = IdentityGallery(min_gap_sim=0.92, max_templates=12, top_k=3, dim=512)

    # Tao 2 vector co do tuong dong rat cao (~0.90)
    base = make_unit_vector(dim=512, seed=42)
    noise1 = make_unit_vector(dim=512, seed=43)
    noise2 = make_unit_vector(dim=512, seed=44)

    v_c = base * 0.95 + noise1 * 0.312
    v_c /= np.linalg.norm(v_c)

    v_d = base * 0.95 + noise2 * 0.312
    v_d /= np.linalg.norm(v_d)

    gallery.enroll("id_C", [v_c], label="Person C", source="disk")
    gallery.enroll("id_D", [v_d], label="Person D", source="disk")

    # Truy van chinh la vector trung gian
    query = (v_c + v_d) / np.linalg.norm(v_c + v_d)
    best_id, score, margin = gallery.match(query)

    assert best_id in ("id_C", "id_D")
    assert score > 0.9
    # Margin phai rat thap do hai nguoi co score gan bang nhau
    assert margin < 0.05


# ---------------------------------------------------------------------------
# Test 3: select_diverse khong chon mau gan trung
# ---------------------------------------------------------------------------
def test_03_select_diverse_filters_near_duplicates():
    v0 = make_unit_vector(dim=512, seed=10)
    # v1 gan giong het v0 (cosine ~ 0.999 > 0.92)
    v1 = v0 + 0.01 * make_unit_vector(dim=512, seed=11)
    v1 /= np.linalg.norm(v1)

    # v2 khac biet hoan toan
    v2 = make_unit_vector(dim=512, seed=20)
    # v3 gan giong het v2 (cosine ~ 0.999 > 0.92)
    v3 = v2 + 0.01 * make_unit_vector(dim=512, seed=21)
    v3 /= np.linalg.norm(v3)

    samples = [v0, v1, v2, v3]

    # Goi qua phuong thuc static
    selected = IdentityGallery.select_diverse(samples, max_n=12, min_gap_sim=0.92)

    # Chi co v0 va v2 duoc giu, v1 va v3 bi tu choi do trung lap
    assert len(selected) == 2
    sim_between_selected = float(np.dot(selected[0], selected[1]))
    assert sim_between_selected <= 0.92


# ---------------------------------------------------------------------------
# Test 4: Claim doc quyen, release roi claim lai duoc
# ---------------------------------------------------------------------------
def test_04_claim_exclusive_release_and_reclaim():
    gallery = IdentityGallery(dim=512)
    v = make_unit_vector(dim=512, seed=1)
    gallery.enroll("id_target", [v], label="Target 1", source="manual_lock")

    # Track 10 claim thanh cong
    assert gallery.claim("id_target", track_id=10) is True

    # Track 20 claim cung id_target -> phai bi tu choi vi dang duoc giu boi Track 10
    assert gallery.claim("id_target", track_id=20) is False

    # Track 10 claim lai chinh id_target -> True (cung track claim lai)
    assert gallery.claim("id_target", track_id=10) is True

    # Kiem tra list_identities xem claim hien tai
    infos = gallery.list_identities()
    assert len(infos) == 1
    assert infos[0]["claim"] == 10

    # Nha claim cua Track 10
    assert gallery.release_track(track_id=10) is True

    # Kiem tra claim da ve None
    infos_after = gallery.list_identities()
    assert infos_after[0]["claim"] is None

    # Track 20 bay gio co the claim duoc
    assert gallery.claim("id_target", track_id=20) is True
    assert gallery.list_identities()[0]["claim"] == 20


# ---------------------------------------------------------------------------
# Test 5: add_template tu choi mau trung va dung o 12 ma fixed con nguyen
# ---------------------------------------------------------------------------
def test_05_add_template_rejects_duplicates_and_caps_at_12_preserving_fixed():
    gallery = IdentityGallery(min_gap_sim=0.92, max_templates=12, top_k=3, dim=512)

    # 1. Enroll voi 3 mau fixed ban dau
    f0 = np.zeros(512, dtype=np.float32); f0[0] = 1.0
    f1 = np.zeros(512, dtype=np.float32); f1[1] = 1.0
    f2 = np.zeros(512, dtype=np.float32); f2[2] = 1.0
    fixed_samples = [f0, f1, f2]

    assert gallery.enroll("id_user", fixed_samples, label="Test User") is True
    assert len(gallery.get("id_user")["templates"]) == 3

    # 2. Tu choi mau trung (sim > 0.92 voi f0)
    f0_dup = f0.copy()
    f0_dup[3] = 0.05
    f0_dup /= np.linalg.norm(f0_dup)
    ok, reason = gallery.add_template("id_user", f0_dup)
    assert ok is False
    assert "redundant" in reason
    assert len(gallery.get("id_user")["templates"]) == 3

    # 3. Them 9 mau adaptive da dang (sim voi f0 >= 0.45 nhung sim voi nhau < 0.92)
    adaptive_added = []
    for idx in range(9):
        basis = np.zeros(512, dtype=np.float32)
        basis[10 + idx] = 1.0
        vec = 0.60 * f0 + 0.80 * basis
        vec /= np.linalg.norm(vec)
        ok, reason = gallery.add_template("id_user", vec)
        assert ok is True
        assert reason == "added"
        adaptive_added.append(vec)

    user_info = gallery.get("id_user")
    assert user_info["n_templates"] == 12

    # Kiem tra 3 mau dau tien la fixed, 9 mau sau la adaptive
    raw_templates = gallery._identities["id_user"]["templates"]
    assert all(t["fixed"] is True for t in raw_templates[:3])
    assert all(t["fixed"] is False for t in raw_templates[3:])

    # 4. Them tiep mau adaptive thu 10 (vuot qua 12 mau)
    basis_extra = np.zeros(512, dtype=np.float32)
    basis_extra[50] = 1.0
    vec_extra = 0.60 * f0 + 0.80 * basis_extra
    vec_extra /= np.linalg.norm(vec_extra)
    ok, reason = gallery.add_template("id_user", vec_extra)
    assert ok is True
    assert "added_evicted_oldest_adaptive" in reason

    # Tong so mau van phai DUNG LAI O 12
    raw_after = gallery._identities["id_user"]["templates"]
    assert len(raw_after) == 12

    # 3 mau fixed ban dau van phai CON NGUYEN VEN
    assert raw_after[0]["fixed"] is True
    assert np.allclose(raw_after[0]["feat"], f0)
    assert raw_after[1]["fixed"] is True
    assert np.allclose(raw_after[1]["feat"], f1)
    assert raw_after[2]["fixed"] is True
    assert np.allclose(raw_after[2]["feat"], f2)

    # Mau adaptive cu nhat (adaptive_added[0] o index 10) da bi loai bo
    assert not any(np.allclose(t["feat"], adaptive_added[0]) for t in raw_after)
    # Mau moi nhat da duoc them vao cuoi
    assert np.allclose(raw_after[-1]["feat"], vec_extra)


# ---------------------------------------------------------------------------
# Test 6: Feature NaN/Inf/zero/sai dim khong crash
# ---------------------------------------------------------------------------
def test_06_invalid_features_nan_inf_zero_wrong_dim_no_crash():
    gallery = IdentityGallery(dim=512)
    v = make_unit_vector(dim=512, seed=99)
    gallery.enroll("id_safe", [v], label="Safe")

    # match voi cac feature khong hop le -> tra ve (None, 0.0, 0.0), khong crash
    assert gallery.match(None) == (None, 0.0, 0.0)
    assert gallery.match(np.full(512, np.nan, dtype=np.float32)) == (None, 0.0, 0.0)
    assert gallery.match(np.full(512, np.inf, dtype=np.float32)) == (None, 0.0, 0.0)
    assert gallery.match(np.zeros(512, dtype=np.float32)) == (None, 0.0, 0.0)
    assert gallery.match(np.ones(256, dtype=np.float32)) == (None, 0.0, 0.0)

    # add_template voi feature khong hop le -> tra ve (False, "invalid_feature")
    ok, reason = gallery.add_template("id_safe", np.full(512, np.nan))
    assert ok is False
    assert reason == "invalid_feature"

    ok, reason = gallery.add_template("id_safe", np.zeros(512))
    assert ok is False
    assert reason == "invalid_feature"


# ---------------------------------------------------------------------------
# Test 7: clear("manual_lock") giu lai identity "disk"
# ---------------------------------------------------------------------------
def test_07_clear_manual_lock_retains_disk_identities():
    gallery = IdentityGallery(dim=512)
    v1 = make_unit_vector(dim=512, seed=1)
    v2 = make_unit_vector(dim=512, seed=2)
    v3 = make_unit_vector(dim=512, seed=3)

    gallery.enroll("id_m1", [v1], source="manual_lock")
    gallery.enroll("id_m2", [v2], source="manual_lock")
    gallery.enroll("id_disk1", [v3], source="disk")

    assert len(gallery) == 3

    # Xoa chi cac identity do manual_lock
    deleted_count = gallery.clear(source="manual_lock")
    assert deleted_count == 2

    # Chi con lai id_disk1
    assert len(gallery) == 1
    ids = [item["id"] for item in gallery.list_identities()]
    assert ids == ["id_disk1"]

    # Xoa toan bo
    gallery.clear(source=None)
    assert len(gallery) == 0


# ---------------------------------------------------------------------------
# Test 8: remove va release_track don sach claims
# ---------------------------------------------------------------------------
def test_08_remove_and_release_track_cleans_claims():
    gallery = IdentityGallery(dim=512)
    v = make_unit_vector(dim=512, seed=7)
    gallery.enroll("id_clean", [v])
    gallery.claim("id_clean", track_id=999)

    assert gallery.get("id_clean")["claim"] == 999

    # Xoa identity -> claim cung phai duoc don dep
    assert gallery.remove("id_clean") is True
    assert gallery.get("id_clean") is None
    # Track 999 khong con giu claim nao nua
    assert gallery.release_track(999) is False


# ---------------------------------------------------------------------------
# Test 9: Da luong (concurrent access) voi match, claim, add_template, enroll, remove
# ---------------------------------------------------------------------------
def test_09_concurrent_threads_access():
    gallery = IdentityGallery(dim=512, min_fixed_sim=0.45)
    base = make_unit_vector(dim=512, seed=100)
    gallery.enroll("id_shared", [base])

    errors = []

    def worker_matcher():
        try:
            for _ in range(50):
                query = make_unit_vector(dim=512)
                gallery.match(query)
        except Exception as e:
            errors.append(e)

    def worker_claimer(tid):
        try:
            for _ in range(50):
                gallery.claim("id_shared", track_id=tid)
                gallery.release_track(track_id=tid)
        except Exception as e:
            errors.append(e)

    def worker_adder():
        try:
            for i in range(50):
                n = make_unit_vector(dim=512)
                v = 0.6 * base + 0.8 * n
                v /= np.linalg.norm(v)
                gallery.add_template("id_shared", v)
        except Exception as e:
            errors.append(e)

    def worker_enroll_remove():
        try:
            for i in range(30):
                tid = f"temp_id_{i}"
                vec = make_unit_vector(dim=512)
                gallery.enroll(tid, [vec], overwrite=True)
                gallery.remove(tid)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker_matcher),
        threading.Thread(target=worker_matcher),
        threading.Thread(target=worker_claimer, args=(1,)),
        threading.Thread(target=worker_claimer, args=(2,)),
        threading.Thread(target=worker_adder),
        threading.Thread(target=worker_enroll_remove),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Test 10: add_template tu choi mau khong giong fixed
# ---------------------------------------------------------------------------
def test_10_add_template_rejects_dissimilar_to_fixed():
    gallery = IdentityGallery(min_fixed_sim=0.45, dim=512)
    f0 = np.zeros(512, dtype=np.float32)
    f0[0] = 1.0
    gallery.enroll("id_fixed_test", [f0])

    # Vector truc giao voi f0 (cosine sim = 0.0 < 0.45)
    v_dissimilar = np.zeros(512, dtype=np.float32)
    v_dissimilar[1] = 1.0

    ok, reason = gallery.add_template("id_fixed_test", v_dissimilar)
    assert ok is False
    assert reason.startswith("not_similar_to_fixed_")
    assert "0.000" in reason


# ---------------------------------------------------------------------------
# Test 11: enroll trung id khong ghi de mac dinh (overwrite=False)
# ---------------------------------------------------------------------------
def test_11_enroll_duplicate_id_does_not_overwrite_by_default():
    gallery = IdentityGallery(dim=512)
    v1 = make_unit_vector(dim=512, seed=1)
    v2 = make_unit_vector(dim=512, seed=2)

    # Enroll lan 1
    assert gallery.enroll("id_dup", [v1], label="Original", overwrite=False) is True
    assert gallery.get("id_dup")["label"] == "Original"
    assert np.allclose(gallery.get("id_dup")["templates"][0], v1)

    # Enroll lan 2 voi overwrite=False (mac dinh) -> bi tu choi va giu nguyen du lieu cu
    assert gallery.enroll("id_dup", [v2], label="Overwritten?", overwrite=False) is False
    assert gallery.get("id_dup")["label"] == "Original"
    assert np.allclose(gallery.get("id_dup")["templates"][0], v1)

    # Enroll voi overwrite=True -> ghi de thanh cong
    assert gallery.enroll("id_dup", [v2], label="Updated", overwrite=True) is True
    assert gallery.get("id_dup")["label"] == "Updated"
    assert np.allclose(gallery.get("id_dup")["templates"][0], v2)


# ---------------------------------------------------------------------------
# Test 12: claim identity khong ton tai tra False
# ---------------------------------------------------------------------------
def test_12_claim_non_existent_identity_returns_false():
    gallery = IdentityGallery(dim=512)
    assert gallery.claim("ghost_identity", track_id=42) is False


# ---------------------------------------------------------------------------
# Test 13: track claim identity B sau A thi A duoc nha
# ---------------------------------------------------------------------------
def test_13_track_claiming_b_after_a_releases_a():
    gallery = IdentityGallery(dim=512)
    v_a = make_unit_vector(dim=512, seed=3)
    v_b = make_unit_vector(dim=512, seed=4)

    gallery.enroll("id_A", [v_a])
    gallery.enroll("id_B", [v_b])

    # Track 10 claim A
    assert gallery.claim("id_A", track_id=10) is True
    assert gallery.get("id_A")["claim"] == 10

    # Track 10 chuyen sang claim B
    assert gallery.claim("id_B", track_id=10) is True
    assert gallery.get("id_B")["claim"] == 10

    # Kiem tra A da duoc nha claim (claim cua A ve None)
    assert gallery.get("id_A")["claim"] is None

    # Track 20 bay gio co the claim A
    assert gallery.claim("id_A", track_id=20) is True
    assert gallery.get("id_A")["claim"] == 20


# ---------------------------------------------------------------------------
# Test 14: sua mang tu get() khong doi RAM
# ---------------------------------------------------------------------------
def test_14_mutating_array_from_get_does_not_mutate_ram():
    gallery = IdentityGallery(dim=512)
    v = make_unit_vector(dim=512, seed=5)
    gallery.enroll("id_immut", [v])

    info = gallery.get("id_immut")
    original_val = float(info["templates"][0][0])

    # Sua truc tiep tren array duoc tra ve tu get()
    info["templates"][0][0] += 999.0

    # Lay lai tu gallery va kiem tra gia tri trong RAM van khong he doi
    info_fresh = gallery.get("id_immut")
    assert np.isclose(info_fresh["templates"][0][0], original_val)


# ---------------------------------------------------------------------------
# Test 15: select_diverse tra mang (0, dim) dung dim dau vao khi rong
# ---------------------------------------------------------------------------
def test_15_select_diverse_returns_correct_dim_when_empty():
    # Mang rong voi dim=256
    empty_256 = np.empty((0, 256), dtype=np.float32)
    res = IdentityGallery.select_diverse(empty_256)
    assert res.shape == (0, 256)

    # None mac dinh tra ve dim=512
    res_none = IdentityGallery.select_diverse(None)
    assert res_none.shape == (0, 512)


# ---------------------------------------------------------------------------
# Test 16: new_id cap id moi va bo qua id da ton tai
# ---------------------------------------------------------------------------
def test_16_new_id_generation_and_skipping():
    gallery = IdentityGallery(dim=512)
    # Mac dinh khoi dau la P1
    assert gallery.new_id() == "P1"

    v = make_unit_vector(dim=512)
    gallery.enroll("P1", [v])
    assert gallery.new_id() == "P2"

    gallery.enroll("P2", [v])
    assert gallery.new_id() == "P3"

    # Neu da co P4 (vi du load tu dia vao truoc), new_id van tra P3
    gallery.enroll("P4", [v])
    assert gallery.new_id() == "P3"

    # Prefix tuy chinh
    assert gallery.new_id(prefix="TARGET_") == "TARGET_1"
