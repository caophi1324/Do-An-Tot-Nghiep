#!/usr/bin/env python3
"""
test_identity_recognition.py - Test suite for identity re-recognition and claim management (Bước 5b & 5b.1).
Covers:
1. Streak progression & assignment upon reaching identity_recog_streak (5).
2. Deviation (below_thr, low_margin, other_identity) resets streak.
3. Interval throttle does NOT reset streak.
4. Race condition: two tracks reaching streak, first to claim wins, second is denied without resetting streak.
5. Stale claim stolen when holder is deleted from tracked_faces.
6. Stale claim stolen when holder has lost >= identity_steal_lost_frames (5).
7. Active holder denies claim (claim_denied), streak preserved.
8. Locked holder with lost <= max_grace_frames (25) is NOT stolen; stolen only when lost > max_grace_frames.
9. Track that is currently locked is bypassed by _identity_observe.
10. Track with active swap_votes is bypassed by _identity_observe.
11. Track already assigned an identity_id is NOT reassigned, logs WARNING conflict on mismatch (throttled 5s).
12. locked_face_id=None still matches and assigns.
13. Verify recognizer.feature() and extract_feat are NOT called additionally.
14. Verify update() returns tuples with exactly length 5: (x1, y1, x2, y2, tid).
15. Verify identity_labels is a brand new dict reference after every update().
16. Concurrency lock test: a thread holds rec_lock (with an Event), another thread calls
    face_tracker.clear(rec_lock=rec_lock) and remove_track, must complete within 2.0s without deadlocking.
17. Diagnostic log: no_feature logged once when track age >= 5.0s, gallery not empty, and not locked.
18. In _verify_and_swap: swap resets id_streak and id_candidate for both swapped tracks.
19. (6a) Tích hợp qua update(): lặp update() gán identity, identity_labels có nhãn, tuple 5 phần tử.
20. (6a) Hook _identity_observe được gọi từ cả Step 4 và Step 6.
21. (6b) Nhả claim từ ngoài: update() kế tiếp xóa identity_id và nhãn, log identity_dropped.
22. (6c) claim_stolen xóa nhãn của holder còn sống trong tracked_faces.
23. (6d) claim_denied bị throttle 5 giây.
24. (6e) Track đang lock có claim thì có nhãn trong identity_labels.
25. (6f) update() với detection không gọi recognizer.feature/extract_feat nhiều hơn số lần các điểm trích hiện có.
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
from person_id_tracker import PersonIDTracker
from identity_gallery import IdentityGallery
from face_utils import FaceGallery


class MockRecognizer:
    def __init__(self, dim=512):
        self.dim = dim
        self.feature_call_count = 0

    def match(self, f1, f2, mode=0):
        v1 = np.asarray(f1, dtype=np.float32).flatten()
        v2 = np.asarray(f2, dtype=np.float32).flatten()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def feature(self, *args, **kwargs):
        self.feature_call_count += 1
        return np.zeros(self.dim, dtype=np.float32)


class MockDetector:
    pass


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class TestIdentityRecognition(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dim = 512

        # Create two orthogonal base vectors for P1 and P2
        self.vec1 = np.zeros(self.dim, dtype=np.float32)
        self.vec1[0] = 1.0

        self.vec2 = np.zeros(self.dim, dtype=np.float32)
        self.vec2[1] = 1.0

        self.recognizer = MockRecognizer(dim=self.dim)
        self.detector = MockDetector()

        self.gallery = IdentityGallery(dim=self.dim)
        self.gallery.enroll("P1", [self.vec1], label="Alice", source="manual_lock")
        self.gallery.enroll("P2", [self.vec2], label="Bob", source="manual_lock")

        self.orig_recognition = dict(settings.recognition)
        self.orig_tracking = dict(settings.tracking)
        settings.recognition["identity_enabled"] = True
        settings.recognition["identity_recog_thr"] = 0.50
        settings.recognition["identity_recog_margin"] = 0.10
        settings.recognition["identity_recog_streak"] = 5
        settings.recognition["identity_recog_interval_sec"] = 0.3
        settings.recognition["identity_steal_lost_frames"] = 5
        settings.tracking["max_grace_frames"] = 25

        # Cô lập id_logger: patch thẳng person_id_tracker.id_logger và tracking.id_logger
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
        settings.tracking.clear()
        settings.tracking.update(self.orig_tracking)

    def _create_tracker(self, max_lost=10):
        return PersonIDTracker(
            self.detector,
            self.recognizer,
            max_lost=max_lost,
            identity_gallery=self.gallery,
        )

    def _add_track(self, tracker, tid, bbox=(10, 10, 50, 50), lost=0, age=0.0):
        now = time.monotonic()
        tracker.tracked_faces[tid] = {
            "bbox": bbox,
            "lost": lost,
            "t": now - age,
            "vx": 0.0,
            "vy": 0.0,
            "gallery": FaceGallery(),
            "first_seen_time": now - age,
            "n_feats": 0,
            "identity_id": None,
            "id_candidate": None,
            "id_streak": 0,
            "last_match_time": 0.0,
            "identity_label": None,
        }
        return tracker.tracked_faces[tid]

    # --- Test 1: Đủ streak thì gán ---
    def test_01_streak_reaches_target_assigns_identity(self):
        """Khi quan sát đủ streak_target (5 lần) đạt ngưỡng, identity được gán và claim thành công."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        sim_vec = self.vec1.copy()  # Cosine sim with P1 is 1.0, with P2 is 0.0

        for i in range(1, 6):
            now = 100.0 + i * 0.5  # intervals > 0.3s
            tracker._identity_observe(tid=1, feat=sim_vec, now=now, locked_face_id=None)
            if i < 5:
                self.assertEqual(tracker.tracked_faces[1]["id_streak"], i)
                self.assertEqual(tracker.tracked_faces[1]["id_candidate"], "P1")
                self.assertIsNone(tracker.tracked_faces[1]["identity_id"])
            else:
                self.assertEqual(tracker.tracked_faces[1]["id_streak"], 5)
                self.assertEqual(tracker.tracked_faces[1]["identity_id"], "P1")
                self.assertEqual(tracker.tracked_faces[1]["identity_label"], "Alice")
                self.assertEqual(self.gallery.get("P1")["claim"], 1)

        # Check logs (bao gồm score và margin)
        assign_logs = [r for r in self.log_handler.records if "assign track=1 id=P1 label=Alice" in r]
        self.assertEqual(len(assign_logs), 1)
        self.assertIn("score=1.000", assign_logs[0])
        self.assertIn("margin=1.000", assign_logs[0])

    # --- Test 2: Một lần lệch thì reset streak ---
    def test_02_streak_resets_on_deviation(self):
        """Lệch do below_thr, low_margin, hoặc other_identity đều reset streak."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        # 2a: below_thr
        # Bước đầu đạt streak 2 với P1
        tracker._identity_observe(tid=1, feat=self.vec1, now=10.0)
        tracker._identity_observe(tid=1, feat=self.vec1, now=10.5)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 2)

        # Quan sát vector ngẫu nhiên không giống P1 hay P2 (score < 0.50)
        rand_vec = np.zeros(self.dim, dtype=np.float32)
        rand_vec[10] = 1.0
        tracker._identity_observe(tid=1, feat=rand_vec, now=11.0)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertIsNone(tracker.tracked_faces[1]["id_candidate"])
        self.assertTrue(any("reason=below_thr" in r for r in self.log_handler.records))

        # 2b: low_margin
        # Vector nằm giữa P1 và P2 (sim(P1)=cos(pi/4)=0.707, sim(P2)=0.707 -> margin=0.0 < 0.10)
        mid_vec = (self.vec1 + self.vec2) / np.sqrt(2.0)
        tracker._identity_observe(tid=1, feat=self.vec1, now=12.0)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 1)
        tracker._identity_observe(tid=1, feat=mid_vec, now=12.5)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertTrue(any("reason=low_margin" in r for r in self.log_handler.records))

        # 2c: other_identity
        tracker._identity_observe(tid=1, feat=self.vec1, now=13.0)
        tracker._identity_observe(tid=1, feat=self.vec1, now=13.5)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 2)
        # Bất ngờ khớp P2 mạnh (score=1.0, margin=1.0)
        tracker._identity_observe(tid=1, feat=self.vec2, now=14.0)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 1)
        self.assertEqual(tracker.tracked_faces[1]["id_candidate"], "P2")
        self.assertTrue(any("reason=other_identity" in r for r in self.log_handler.records))

    # --- Test 3: Throttle không reset streak ---
    def test_03_throttle_does_not_reset_streak(self):
        """Quan sát dồn dập trong khoảng < identity_recog_interval_sec (0.3s) bị throttle bỏ qua, KHÔNG reset streak."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        tracker._identity_observe(tid=1, feat=self.vec1, now=10.0)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 1)
        self.assertEqual(tracker.tracked_faces[1]["n_feats"], 1)

        # Quan sát sau 0.1s (< 0.3s)
        tracker._identity_observe(tid=1, feat=self.vec1, now=10.1)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 1)  # Giữ nguyên streak
        self.assertEqual(tracker.tracked_faces[1]["n_feats"], 2)   # Vẫn đếm n_feats

        # Quan sát sau 0.35s tính từ 10.0 (tức 10.35 >= 10.0 + 0.3)
        tracker._identity_observe(tid=1, feat=self.vec1, now=10.35)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 2)
        self.assertEqual(tracker.tracked_faces[1]["n_feats"], 3)

    # --- Test 4: Hai track cùng đạt, chỉ track claim trước được giữ ---
    def test_04_race_condition_two_tracks_reach_streak(self):
        """Track nào đạt streak và gọi claim trước sẽ được gán; track sau nhận claim_denied và giữ streak."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        self._add_track(tracker, tid=2)

        for i in range(1, 5):
            tracker._identity_observe(tid=1, feat=self.vec1, now=10.0 + i * 0.5)
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5)

        # Cả hai đang có streak 4
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 4)
        self.assertEqual(tracker.tracked_faces[2]["id_streak"], 4)

        # Track 1 gọi lần 5 trước -> claim P1 thành công
        tracker._identity_observe(tid=1, feat=self.vec1, now=13.0)
        self.assertEqual(tracker.tracked_faces[1]["identity_id"], "P1")
        self.assertEqual(self.gallery.get("P1")["claim"], 1)

        # Track 2 gọi lần 5 sau -> holder 1 đang active (lost=0) -> claim_denied
        tracker._identity_observe(tid=2, feat=self.vec1, now=13.1)
        self.assertIsNone(tracker.tracked_faces[2]["identity_id"])
        self.assertEqual(tracker.tracked_faces[2]["id_streak"], 5)  # Giữ streak
        self.assertTrue(any("claim_denied id=P1 track=2 held_by=1" in r for r in self.log_handler.records))

    # --- Test 5: Holder không còn trong tracked_faces thì claim chuyển (stolen) ---
    def test_05_stale_claim_stolen_holder_not_in_tracked_faces(self):
        """Nếu holder cũ không còn trong tracked_faces, claim bị xem là stale và được chuyển cho track mới."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=2)

        # Gán claim cho track 999 (không tồn tại trong tracked_faces)
        self.gallery.claim("P1", track_id=999)
        self.assertEqual(self.gallery.get("P1")["claim"], 999)

        # Track 2 tích lũy streak 5
        for i in range(1, 6):
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5)

        self.assertEqual(tracker.tracked_faces[2]["identity_id"], "P1")
        self.assertEqual(self.gallery.get("P1")["claim"], 2)
        stolen_logs = [r for r in self.log_handler.records if "claim_stolen from=999 to=2 id=P1" in r]
        self.assertEqual(len(stolen_logs), 1)
        self.assertIn("score=1.000", stolen_logs[0])
        self.assertIn("margin=1.000", stolen_logs[0])

    # --- Test 6: Holder lost >= identity_steal_lost_frames thì claim chuyển ---
    def test_06_stale_claim_stolen_holder_lost_exceeds_threshold(self):
        """Track 1 là holder bình thường nhưng lost=5 (>= identity_steal_lost_frames=5) -> claim bị cướp sang track 2."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1, lost=5)
        self._add_track(tracker, tid=2, lost=0)

        self.gallery.claim("P1", track_id=1)
        tracker.tracked_faces[1]["identity_id"] = "P1"

        for i in range(1, 6):
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5)

        self.assertEqual(tracker.tracked_faces[2]["identity_id"], "P1")
        self.assertEqual(self.gallery.get("P1")["claim"], 2)
        stolen_logs = [r for r in self.log_handler.records if "claim_stolen from=1 to=2 id=P1" in r]
        self.assertEqual(len(stolen_logs), 1)
        self.assertIn("score=1.000", stolen_logs[0])
        self.assertIn("margin=1.000", stolen_logs[0])

    # --- Test 7: Holder còn sống thì claim_denied ---
    def test_07_claim_denied_when_holder_alive(self):
        """Track 1 là holder với lost=2 (< 5) -> track 2 bị từ chối claim (claim_denied) và giữ nguyên streak."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1, lost=2)
        self._add_track(tracker, tid=2, lost=0)

        self.gallery.claim("P1", track_id=1)

        for i in range(1, 6):
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5)

        self.assertIsNone(tracker.tracked_faces[2]["identity_id"])
        self.assertEqual(tracker.tracked_faces[2]["id_streak"], 5)
        self.assertEqual(self.gallery.get("P1")["claim"], 1)
        self.assertTrue(any("claim_denied id=P1 track=2 held_by=1" in r for r in self.log_handler.records))

    # --- Test 8: Locked holder: lost <= max_grace_frames không chuyển, lost > max_grace_frames thì chuyển ---
    def test_08_locked_holder_grace_frames_threshold(self):
        """Nếu holder là locked_face_id:
        - lost <= max_grace_frames (25): KHÔNG bị cướp claim.
        - lost > max_grace_frames (26): BỊ cướp claim."""
        # 8a: lost = 25 (<= 25)
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1, lost=25)
        self._add_track(tracker, tid=2, lost=0)
        self.gallery.claim("P1", track_id=1)

        for i in range(1, 6):
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5, locked_face_id=1)

        self.assertIsNone(tracker.tracked_faces[2]["identity_id"])
        self.assertEqual(self.gallery.get("P1")["claim"], 1)
        self.assertTrue(any("claim_denied id=P1 track=2 held_by=1" in r for r in self.log_handler.records))

        # 8b: lost = 26 (> 25)
        tracker.tracked_faces[1]["lost"] = 26
        self.log_handler.records.clear()
        tracker._identity_observe(tid=2, feat=self.vec1, now=20.0, locked_face_id=1)

        self.assertEqual(tracker.tracked_faces[2]["identity_id"], "P1")
        self.assertEqual(self.gallery.get("P1")["claim"], 2)
        self.assertTrue(any("claim_stolen from=1 to=2 id=P1" in r for r in self.log_handler.records))

    # --- Test 9: Track locked bị bỏ qua ---
    def test_09_locked_track_bypassed_by_identity_observe(self):
        """Track đang khóa (tid == locked_face_id) bị bỏ qua hoàn toàn trong _identity_observe."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        tracker._identity_observe(tid=1, feat=self.vec1, now=10.0, locked_face_id=1)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertEqual(tracker.tracked_faces[1]["last_match_time"], 0.0)
        self.assertEqual(len(self.log_handler.records), 0)

    # --- Test 10: Có swap_votes thì bỏ qua ---
    def test_10_swap_votes_bypasses_identity_observe(self):
        """Nếu track nằm trong swap_votes, bỏ qua quan sát để tránh nhận diện sai khi đang nghi ngờ swap."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        tracker.swap_votes[(1, 2)] = 1

        tracker._identity_observe(tid=1, feat=self.vec1, now=10.0)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertEqual(tracker.tracked_faces[1]["last_match_time"], 0.0)

    # --- Test 11: Track đã có identity không bị đổi và chỉ ghi WARNING khi lệch ---
    def test_11_already_assigned_track_logs_warning_on_conflict_no_change(self):
        """Track đã có identity_id="P1", khi quan sát thấy P2 thì KHÔNG đổi identity, chỉ ghi WARNING (throttled 5s)."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        tracker.tracked_faces[1]["identity_id"] = "P1"
        self.gallery.claim("P1", track_id=1)

        # Quan sát vector P2 (mismatch)
        tracker._identity_observe(tid=1, feat=self.vec2, now=10.0)
        self.assertEqual(tracker.tracked_faces[1]["identity_id"], "P1")
        self.assertTrue(any("[IDENTITY] conflict track=1 has=P1 match=P2" in r for r in self.log_handler.records))

        # Quan sát lại P2 sau 1s (< 5s) -> bị throttle warning
        self.log_handler.records.clear()
        tracker._identity_observe(tid=1, feat=self.vec2, now=11.0)
        self.assertEqual(tracker.tracked_faces[1]["identity_id"], "P1")
        self.assertEqual(len(self.log_handler.records), 0)

        # Quan sát lại P2 sau 6s (>= 5s) -> warning lại ghi
        tracker._identity_observe(tid=1, feat=self.vec2, now=17.0)
        self.assertTrue(any("[IDENTITY] conflict track=1 has=P1 match=P2" in r for r in self.log_handler.records))

    # --- Test 12: locked_face_id=None vẫn match ---
    def test_12_locked_face_id_none_matches_and_assigns(self):
        """Khi locked_face_id=None (hệ thống chưa khóa ai), track vẫn nhận diện và gán identity bình thường."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        for i in range(1, 6):
            tracker._identity_observe(tid=1, feat=self.vec1, now=10.0 + i * 0.5, locked_face_id=None)

        self.assertEqual(tracker.tracked_faces[1]["identity_id"], "P1")
        self.assertEqual(self.gallery.get("P1")["claim"], 1)

    # --- Test 13: recognizer.feature và extract_feat không bị gọi thêm ---
    def test_13_no_additional_feature_extraction(self):
        """_identity_observe chỉ dùng feature đã trích xuất sẵn, tuyệt đối không gọi thêm recognizer.feature()."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)

        initial_count = self.recognizer.feature_call_count
        for i in range(1, 6):
            tracker._identity_observe(tid=1, feat=self.vec1, now=10.0 + i * 0.5)

        self.assertEqual(self.recognizer.feature_call_count, initial_count)
        self.assertEqual(self.recognizer.feature_call_count, 0)

    # --- Test 14: update() trả tuple đúng 5 phần tử ---
    def test_14_update_returns_tuples_of_exactly_length_5(self):
        """update() luôn trả về danh sách các tuple/list có chính xác 5 phần tử (x1, y1, x2, y2, tid)."""
        tracker = self._create_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test case empty detections
        res_empty = tracker.update([], frame)
        self.assertEqual(res_empty, [])

        # Test case with detections
        dets = [
            (50, 50, 150, 150),
            (200, 200, 300, 300),
        ]
        with patch("person_id_tracker._safe_extract_feat", return_value=(None, None, 0.0, (0, 0, 0, 0))):
            res = tracker.update(dets, frame)

        self.assertEqual(len(res), 2)
        for item in res:
            self.assertEqual(len(item), 5, f"Expected 5 elements, got {len(item)}: {item}")
            x1, y1, x2, y2, tid = item
            self.assertIsInstance(tid, int)

    # --- Test 15: identity_labels là dict mới sau mỗi update ---
    def test_15_identity_labels_is_new_dict_reference_after_update(self):
        """identity_labels luôn được gán lại nguyên khối (dict mới) sau mỗi update(), không sửa tại chỗ."""
        tracker = self._create_tracker()
        initial_labels_id = id(tracker.identity_labels)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update([], frame)
        self.assertNotEqual(id(tracker.identity_labels), initial_labels_id)

        labels_id_after_empty = id(tracker.identity_labels)
        with patch("person_id_tracker._safe_extract_feat", return_value=(None, None, 0.0, (0, 0, 0, 0))):
            tracker.update([(10, 10, 50, 50)], frame)
        self.assertNotEqual(id(tracker.identity_labels), labels_id_after_empty)

    # --- Test 16: Concurrency lock test (face_tracker.clear vs rec_lock) ---
    def test_16_no_deadlock_clear_and_remove_track_under_rec_lock(self):
        """Một thread giữ rec_lock, thread khác gọi clear() và remove_track() phải hoàn thành trong 2s không treo."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        self._add_track(tracker, tid=2)

        rec_lock = threading.RLock()
        rec_lock_held = threading.Event()
        finish_event = threading.Event()

        def lock_holder():
            with rec_lock:
                rec_lock_held.set()
                # Giữ rec_lock cho đến khi finish_event được set
                finish_event.wait(timeout=3.0)

        t_holder = threading.Thread(target=lock_holder)
        t_holder.start()

        # Đợi thread kia nắm được rec_lock
        self.assertTrue(rec_lock_held.wait(timeout=1.0))

        start_time = time.monotonic()
        # Gọi remove_track và clear từ thread chính
        tracker.remove_track(tid=1)
        tracker.clear(rec_lock=rec_lock)
        elapsed = time.monotonic() - start_time

        finish_event.set()
        t_holder.join(timeout=2.0)

        self.assertLess(elapsed, 2.0, "clear() or remove_track() blocked on rec_lock!")
        self.assertEqual(len(tracker.tracked_faces), 0)

    # --- Test 17: Log no_feature cho track age >= 5s không có identity ---
    def test_17_no_feature_log_for_tracks_without_feature(self):
        """Track tồn tại >= 5s không được trích xuất feature và chưa gán identity sẽ ghi log no_feature một lần."""
        tracker = self._create_tracker()
        now = time.monotonic()
        # Track 1 tồn tại 6s trước
        self._add_track(tracker, tid=1, age=6.0)
        self.log_handler.records.clear()

        # Update rỗng
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Để chạy vào nhánh cuối update:
        with patch("person_id_tracker._safe_extract_feat", return_value=(None, None, 0.0, (0, 0, 0, 0))):
            tracker.update([(10, 10, 50, 50)], frame)

        no_feat_logs = [r for r in self.log_handler.records if "no_feature track=1" in r]
        self.assertEqual(len(no_feat_logs), 1)

        # Update lần nữa: không ghi lại (đã có cờ no_feat_logged)
        self.log_handler.records.clear()
        with patch("person_id_tracker._safe_extract_feat", return_value=(None, None, 0.0, (0, 0, 0, 0))):
            tracker.update([(10, 10, 50, 50)], frame)
        no_feat_logs_2 = [r for r in self.log_handler.records if "no_feature track=1" in r]
        self.assertEqual(len(no_feat_logs_2), 0)

    # --- Test 18: _verify_and_swap resets id_streak và id_candidate ---
    def test_18_verify_and_swap_resets_streak_for_both_tracks(self):
        """Khi swap xảy ra trong _verify_and_swap, streak và candidate của cả 2 track bị reset."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        self._add_track(tracker, tid=2)

        tracker.tracked_faces[1]["id_streak"] = 3
        tracker.tracked_faces[1]["id_candidate"] = "P1"
        tracker.tracked_faces[2]["id_streak"] = 4
        tracker.tracked_faces[2]["id_candidate"] = "P2"

        # Gallery 1 has vec1, Gallery 2 has vec2
        # But det 0 (assigned to 1) has vec2 (matches 2), det 1 (assigned to 2) has vec1 (matches 1)
        tracker.tracked_faces[1]["gallery"].add(self.vec1, source="enroll")
        tracker.tracked_faces[2]["gallery"].add(self.vec2, source="enroll")

        # Set swap votes needed = 1 và verify every frame
        settings.recognition["verify_interval_frames"] = 1
        settings.recognition["swap_votes"] = 1
        assign = {0: 1, 1: 2}
        boxes = [(10, 10, 50, 50), (100, 100, 150, 150)]
        crowded = [False, False]
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Mock _safe_extract_feat to return vec2 for det 0 (assigned to track 1, but best matches 2)
        # and vec1 for det 1 (assigned to track 2, but best matches 1)
        def mock_extract(detector, recognizer, img, box, head_ratio, rec_lock):
            if box == boxes[0]:
                return self.vec2, None, 0.9, box
            else:
                return self.vec1, None, 0.9, box

        with patch("person_id_tracker._safe_extract_feat", side_effect=mock_extract):
            swapped = tracker._verify_and_swap(
                assign, boxes, crowded, frame, rec_lock=None, head_ratio=0.45, thres=0.4
            )

        self.assertIn(1, swapped)
        self.assertIn(2, swapped)
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertIsNone(tracker.tracked_faces[1]["id_candidate"])
        self.assertEqual(tracker.tracked_faces[2]["id_streak"], 0)
        self.assertIsNone(tracker.tracked_faces[2]["id_candidate"])

    # --- Test 19: (6a) Tích hợp qua update() gán identity và có nhãn trong identity_labels ---
    def test_19_integration_update_loop_assigns_and_verifies_labels(self):
        """Gọi update() lặp với cùng detection khớp P1, đặt identity_recog_interval_sec=0:
        đủ streak thì gán, identity_labels có {tid: label}, và update() trả tuple đúng 5 phần tử."""
        settings.recognition["identity_recog_interval_sec"] = 0.0
        settings.recognition["identity_recog_streak"] = 5
        settings.recognition["enroll_min_interval_sec"] = 0.0
        settings.recognition["verify_interval_frames"] = 1
        tracker = self._create_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = [(50, 50, 150, 150)]

        with patch("person_id_tracker._safe_extract_feat", return_value=(self.vec1, None, 0.9, (50, 50, 150, 150))):
            res = None
            for _ in range(5):
                res = tracker.update(det, frame)

        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 5)
        tid = res[0][4]
        self.assertEqual(tracker.tracked_faces[tid]["identity_id"], "P1")
        self.assertIn(tid, tracker.identity_labels)
        self.assertEqual(tracker.identity_labels[tid], "Alice")
        self.assertEqual(self.gallery.get("P1")["claim"], tid)

    # --- Test 20: (6a) Hook _identity_observe được gọi từ Step 4 và Step 6 ---
    def test_20_identity_observe_called_from_step4_and_step6(self):
        """Xác nhận _identity_observe được hook từ cả Step 4 (track mới) và Step 6 (_verify_and_swap)."""
        tracker = self._create_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Step 4 (track mới):
        with patch.object(tracker, "_identity_observe", wraps=tracker._identity_observe) as spy_observe:
            with patch("person_id_tracker._safe_extract_feat", return_value=(self.vec1, None, 0.9, (10, 10, 50, 50))):
                tracker.update([(10, 10, 50, 50)], frame)
            # Được gọi cho track mới tạo
            spy_observe.assert_called()
            call_tids = [call.args[0] for call in spy_observe.call_args_list]
            self.assertIn(1, call_tids)

        # 2. Step 6 (_verify_and_swap):
        # Thiết lập track 1 đã có gallery và match
        settings.recognition["verify_interval_frames"] = 1
        settings.recognition["gallery_add_sim"] = 0.5
        settings.recognition["gallery_add_margin"] = 0.10
        tracker.tracked_faces[1]["gallery"].add(self.vec1, source="enroll")

        with patch.object(tracker, "_identity_observe", wraps=tracker._identity_observe) as spy_observe_step6:
            with patch("person_id_tracker._safe_extract_feat", return_value=(self.vec1, None, 0.9, (10, 10, 50, 50))):
                tracker.update([(10, 10, 50, 50)], frame)
            spy_observe_step6.assert_called()

    # --- Test 21: (6b) Nhả claim từ ngoài -> update() kế tiếp xóa identity_id và nhãn ---
    def test_21_external_release_track_drops_identity_on_next_update(self):
        """Khi nhả claim từ ngoài bằng identity_gallery.release_track, update() kế tiếp chạy _sync_identity_state
        xóa identity_id, identity_label và nhãn trong identity_labels, log identity_dropped."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1)
        self.gallery.claim("P1", track_id=1)
        tracker.tracked_faces[1]["identity_id"] = "P1"
        tracker.tracked_faces[1]["identity_label"] = "Alice"
        tracker.tracked_faces[1]["id_streak"] = 5
        tracker.tracked_faces[1]["id_candidate"] = "P1"

        # Update bình thường -> có nhãn
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker._update_identity_labels()
        self.assertEqual(tracker.identity_labels.get(1), "Alice")

        # Nhả claim từ ngoài
        self.gallery.release_track(1)
        self.log_handler.records.clear()

        # Update tiếp theo kích hoạt _sync_identity_state()
        tracker.update([], frame)

        self.assertIsNone(tracker.tracked_faces[1]["identity_id"])
        self.assertIsNone(tracker.tracked_faces[1]["identity_label"])
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertIsNone(tracker.tracked_faces[1]["id_candidate"])
        self.assertNotIn(1, tracker.identity_labels)
        self.assertTrue(any("identity_dropped track=1 id=P1 reason=claim_lost" in r for r in self.log_handler.records))

    # --- Test 22: (6c) claim_stolen xóa nhãn của holder còn sống trong tracked_faces ---
    def test_22_claim_stolen_clears_living_holder_identity_and_labels(self):
        """Khi claim_stolen xảy ra, nếu holder còn sống trong tracked_faces thì xóa sạch identity_id,
        identity_label, id_streak, id_candidate của holder."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1, lost=5)
        self._add_track(tracker, tid=2, lost=0)
        self.gallery.claim("P1", track_id=1)
        tracker.tracked_faces[1]["identity_id"] = "P1"
        tracker.tracked_faces[1]["identity_label"] = "Alice"
        tracker.tracked_faces[1]["id_streak"] = 5
        tracker.tracked_faces[1]["id_candidate"] = "P1"

        # Track 2 đạt streak 5 và claim P1
        for i in range(1, 6):
            tracker._identity_observe(tid=2, feat=self.vec1, now=10.0 + i * 0.5)

        # Holder 1 đã bị xóa sạch thông tin identity
        self.assertIsNone(tracker.tracked_faces[1]["identity_id"])
        self.assertIsNone(tracker.tracked_faces[1]["identity_label"])
        self.assertEqual(tracker.tracked_faces[1]["id_streak"], 0)
        self.assertIsNone(tracker.tracked_faces[1]["id_candidate"])

        # Track 2 nhận identity
        self.assertEqual(tracker.tracked_faces[2]["identity_id"], "P1")
        self.assertEqual(tracker.tracked_faces[2]["identity_label"], "Alice")
        self.assertEqual(self.gallery.get("P1")["claim"], 2)

        # Cập nhật nhãn: chỉ có track 2 có nhãn
        tracker._update_identity_labels()
        self.assertNotIn(1, tracker.identity_labels)
        self.assertEqual(tracker.identity_labels.get(2), "Alice")

    # --- Test 23: (6d) claim_denied bị throttle 5 giây ---
    def test_23_claim_denied_throttled_5_seconds(self):
        """claim_denied chỉ log tối đa 1 lần/5 giây/track."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=1, lost=0)
        self._add_track(tracker, tid=2, lost=0)
        self.gallery.claim("P1", track_id=1)
        tracker.tracked_faces[2]["id_streak"] = 5
        tracker.tracked_faces[2]["id_candidate"] = "P1"

        self.log_handler.records.clear()
        # Lần 1 tại t=10.0: log
        tracker._identity_observe(tid=2, feat=self.vec1, now=10.0)
        denied_logs_1 = [r for r in self.log_handler.records if "claim_denied id=P1 track=2 held_by=1" in r]
        self.assertEqual(len(denied_logs_1), 1)

        # Lần 2 tại t=11.0 (< 10.0 + 5.0): bị throttle
        self.log_handler.records.clear()
        tracker._identity_observe(tid=2, feat=self.vec1, now=11.0)
        denied_logs_2 = [r for r in self.log_handler.records if "claim_denied id=P1 track=2 held_by=1" in r]
        self.assertEqual(len(denied_logs_2), 0)

        # Lần 3 tại t=15.5 (>= 10.0 + 5.0): log lại
        self.log_handler.records.clear()
        tracker._identity_observe(tid=2, feat=self.vec1, now=15.5)
        denied_logs_3 = [r for r in self.log_handler.records if "claim_denied id=P1 track=2 held_by=1" in r]
        self.assertEqual(len(denied_logs_3), 1)

    # --- Test 24: (6e) Track đang lock có claim thì có nhãn trong identity_labels ---
    def test_24_locked_track_with_claim_has_label_in_identity_labels(self):
        """Track đang khóa có claim trong gallery thì tự động có nhãn trong identity_labels sau update."""
        tracker = self._create_tracker()
        self._add_track(tracker, tid=10)
        self.gallery.claim("P1", track_id=10)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.update([], frame, locked_face_id=10)
        self.assertEqual(tracker.identity_labels.get(10), "Alice")

    # --- Test 25: (6f) update() với detection không gọi trích xuất feature nhiều hơn số lần hiện có ---
    def test_25_safe_extract_feat_call_count_bounded(self):
        """update() không gọi recognizer.feature hoặc _safe_extract_feat thừa; số lần gọi không vượt quá số detection."""
        tracker = self._create_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            (50, 50, 150, 150),
            (200, 200, 300, 300),
        ]

        with patch("person_id_tracker._safe_extract_feat", return_value=(self.vec1, None, 0.9, (0, 0, 0, 0))) as mock_extract:
            tracker.update(dets, frame)
            # Với 2 detection mới, Step 4 trích xuất tối đa 2 lần (1 lần cho mỗi detection chưa gán)
            self.assertLessEqual(mock_extract.call_count, len(dets))


if __name__ == "__main__":
    unittest.main()
