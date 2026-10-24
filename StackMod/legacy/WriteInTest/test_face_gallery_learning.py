"""
Bộ kiểm thử toàn diện cho Quản lý CSDL Khuôn Mặt Bằng Ảnh (Face Gallery Image-Backed),
Học Liên Tục (Continuous Learning), Chống Nhiễm Mẫu và Log Chẩn Đoán.
"""
import sys
import os
import time
import unittest
import numpy as np

# Thêm đường dẫn pc_client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

import face_utils
from face_utils import FaceGallery, GalleryItem, iou, is_crowded
from tracking import TrackingManager
from person_id_tracker import PersonIDTracker
from config import settings
from event_logger import id_logger


class MockRecognizer:
    def __init__(self, sim_map=None):
        self.sim_map = sim_map or {}

    def match(self, emb1, emb2, mode):
        # Trả về cosine similarity dựa trên dot product hoặc sim_map
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = np.asarray(emb1, dtype=np.float32).flatten()
        v2 = np.asarray(emb2, dtype=np.float32).flatten()
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            return float(np.dot(v1, v2) / (n1 * n2))
        return 0.0

    def alignCrop(self, frame, face):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        return np.ones(512, dtype=np.float32)


class MockDetector:
    def __init__(self, face_box=True):
        self.face_box = face_box

    def setInputSize(self, size):
        pass

    def detect(self, roi):
        if not self.face_box:
            return True, None
        h, w = roi.shape[:2]
        f = np.zeros((1, 15), dtype=np.float32)
        f[0, 0] = 5
        f[0, 1] = 5
        f[0, 2] = min(60, w - 10)
        f[0, 3] = min(60, h - 10)
        f[0, 14] = 0.95
        return True, f


class TestFaceGalleryLearning(unittest.TestCase):
    def setUp(self):
        settings.recognition["gallery_max"] = 12
        settings.recognition["learn_min_interval_sec"] = 0.05
        settings.recognition["learn_block_sec"] = 0.3
        settings.recognition["learn_enabled"] = True
        settings.recognition["gallery_add_sim"] = 0.55
        settings.recognition["gallery_add_margin"] = 0.10
        settings.recognition["face_threshold"] = 0.38
        settings.recognition["recognition_interval_frames"] = 1
        settings.recognition["profile_dup_sim"] = 0.97

    def test_01_gallery_capacity_and_anchor_immutability(self):
        """1. Gallery không bao giờ vượt quá gallery_max (12) và anchor bất biến."""
        gallery = FaceGallery(gallery_max=12)
        anchor_vec = np.zeros(128, dtype=np.float32)
        anchor_vec[0] = 1.0
        anchor_img = np.full((96, 96, 3), 200, dtype=np.uint8)

        # Nạp anchor
        gallery.add(anchor_vec, img=anchor_img, sim=1.0, source="enroll")
        self.assertFalse(gallery.empty())
        self.assertTrue(np.allclose(gallery.anchor, anchor_vec))
        self.assertIsNotNone(gallery.anchor_item.img)
        self.assertEqual(gallery.anchor_item.img.shape, (96, 96, 3))

        # Nạp 20 mẫu khác nhau
        for i in range(1, 21):
            vec = np.zeros(128, dtype=np.float32)
            vec[i % 128] = float(i)
            vec = vec / np.linalg.norm(vec)
            img = np.full((96, 96, 3), i, dtype=np.uint8)
            gallery.add(vec, img=img, sim=0.8, source="learn")

        snap = gallery.snapshot()
        # Tổng số mẫu trong snapshot không bao giờ vượt gallery_max (12)
        self.assertEqual(len(snap), 12)
        self.assertEqual(len(gallery.items), 11)  # 11 items + 1 anchor = 12 total

        # Anchor tuyệt đối không bị thay đổi hay ghi đè
        self.assertTrue(snap[0]["is_anchor"])
        self.assertTrue(np.allclose(snap[0]["feat"], anchor_vec))
        self.assertEqual(snap[0]["source"], "enroll")
        self.assertTrue(np.array_equal(snap[0]["img"], anchor_img))

    def test_02_prune_most_redundant_when_full(self):
        """2. Khi đầy, mẫu dư thừa nhất (cosine similarity cao nhất với mẫu khác) bị tỉa bỏ."""
        gallery = FaceGallery(gallery_max=4)  # anchor + 3 items

        anc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gallery.add(anc, source="enroll")

        it1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        gallery.add(it1, source="it1")

        it2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        gallery.add(it2, source="it2")

        it3 = np.array([0.001, 0.99999, 0.0], dtype=np.float32)
        it3 = it3 / np.linalg.norm(it3)
        gallery.add(it3, source="it3_dup")

        self.assertEqual(len(gallery.snapshot()), 4)

        # Nạp thêm Item 4: vector phân giác góc mới
        it4 = np.array([0.0, 0.707, 0.707], dtype=np.float32)
        gallery.add(it4, source="it4_new")

        # Số lượng vẫn giữ tối đa 4
        snap = gallery.snapshot()
        self.assertEqual(len(snap), 4)

        sources = [it["source"] for it in snap]
        self.assertIn("enroll", sources)  # Anchor bất biến
        self.assertIn("it2", sources)     # Góc độc nhất không bị tỉa
        self.assertIn("it4_new", sources) # Mẫu mới được thêm

        # Chỉ có thể còn 1 trong 2 mẫu bị trùng nhau (it1 hoặc it3_dup)
        has_it1 = "it1" in sources
        has_it3 = "it3_dup" in sources
        self.assertTrue(has_it1 ^ has_it3, "Chỉ giữ lại 1 trong 2 mẫu bị trùng lặp")

    def test_03_image_and_embedding_synchrony(self):
        """3. Ảnh thumbnail 96x96 và vector embedding luôn đồng bộ 1:1 cả số lượng và thứ tự."""
        gallery = FaceGallery(gallery_max=5)
        for i in range(5):
            feat = np.zeros(64, dtype=np.float32)
            feat[i] = 1.0
            img = np.full((96, 96, 3), fill_value=i * 40, dtype=np.uint8)
            gallery.add(feat, img=img, sim=0.9, source=f"src_{i}")

        snap = gallery.snapshot()
        items_img = gallery.get_items_with_images()

        self.assertEqual(len(snap), 5)
        self.assertEqual(len(items_img), 5)

        for i in range(5):
            self.assertTrue(np.allclose(snap[i]["feat"], items_img[i][0]))
            self.assertEqual(np.argmax(snap[i]["feat"]), i)
            self.assertIsNotNone(snap[i]["img"])
            self.assertEqual(snap[i]["img"].shape, (96, 96, 3))
            self.assertEqual(snap[i]["img"][0, 0, 0], i * 40)
            self.assertEqual(items_img[i][1][0, 0, 0], i * 40)

    def test_04_gallery_anchor_manipulation_and_deletion(self):
        """4. Thao tác đặt làm ảnh gốc (set_anchor) và xóa ảnh (remove_item) thread-safe."""
        gallery = FaceGallery(gallery_max=5)
        f0 = np.array([1.0, 0.0], dtype=np.float32)
        f1 = np.array([0.0, 1.0], dtype=np.float32)
        f2 = np.array([0.707, 0.707], dtype=np.float32)

        gallery.add(f0, source="anc_0")
        gallery.add(f1, source="item_1")
        gallery.add(f2, source="item_2")

        self.assertEqual(len(gallery.snapshot()), 3)
        self.assertTrue(np.allclose(gallery.anchor, f0))

        # Đặt item_2 (index 2) làm anchor mới
        success = gallery.set_anchor_from_index(2)
        self.assertTrue(success)
        self.assertTrue(np.allclose(gallery.anchor, f2))

        snap = gallery.snapshot()
        self.assertEqual(len(snap), 3)
        self.assertEqual(snap[0]["source"], "anchor")
        self.assertTrue(np.allclose(snap[0]["feat"], f2))

        # Xóa item tại index 1
        del_ok = gallery.remove_item_by_index(1)
        self.assertTrue(del_ok)
        self.assertEqual(len(gallery.snapshot()), 2)

    def test_05_cross_contamination_prevention(self):
        """5. Chống nhiễm mẫu: Khi hai người giao nhau (crowded), học liên tục bị đóng băng hoàn toàn."""
        import tracking
        tracker = TrackingManager(MockDetector(face_box=True), MockRecognizer())
        tracker.learn_enabled = True

        tracker.state = "TRACKING"
        tracker.locked_face_id = 1
        tracker.locked_bbox = [100, 100, 200, 300]
        tracker.target_bbox = [100, 100, 200, 300]

        anc_feat = np.zeros(512, dtype=np.float32)
        anc_feat[0] = 1.0
        tracker.gallery.add(anc_feat, sim=1.0, source="enroll")
        initial_gallery_len = len(tracker.gallery.snapshot())

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Hai người đứng đè lên nhau (IoU > 0.25 -> crowded_now = True)
        det_target = [100, 100, 200, 300, 1]
        det_stranger = [120, 110, 220, 310, 2]
        detections = [det_target, det_stranger]

        sample_feat = np.zeros(512, dtype=np.float32)
        sample_feat[0] = 0.85
        sample_feat[1] = 0.52678
        sample_feat = sample_feat / np.linalg.norm(sample_feat)
        sample_img = np.zeros((112, 112, 3), dtype=np.uint8)

        def mock_extract(det, rec, f, box, ratio, lock):
            return sample_feat, sample_img

        orig_extract = tracking.extract_feat
        tracking.extract_feat = mock_extract
        try:
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=1)
            # Gallery không được phép nhận thêm mẫu vì crowded_now = True
            self.assertEqual(len(tracker.gallery.snapshot()), initial_gallery_len)
            self.assertIsNone(tracker.learn_staging)
        finally:
            tracking.extract_feat = orig_extract

    def test_06_learning_blocked_after_mismatch_and_drop(self):
        """6. Dừng học trong thời gian learn_block_sec sau khi có mismatch, swap hoặc drop."""
        import tracking
        tracker = TrackingManager(MockDetector(face_box=True), MockRecognizer())
        tracker.learn_enabled = True
        tracker.learn_block_sec = 0.3  # 300ms block
        tracker.learn_min_interval_sec = 0.05
        tracker.state = "TRACKING"
        tracker.locked_face_id = 1
        tracker.locked_bbox = [100, 100, 200, 300]
        tracker.target_bbox = [100, 100, 200, 300]

        anc_feat = np.zeros(512, dtype=np.float32)
        anc_feat[0] = 1.0
        tracker.gallery.add(anc_feat, sim=1.0, source="enroll")

        # Ghi nhận sự kiện mismatch gây block
        tracker.last_block_time = time.monotonic()
        tracker.learn_staging = None

        good_feat = np.zeros(512, dtype=np.float32)
        good_feat[0] = 0.85
        good_feat[1] = 0.52678
        good_feat = good_feat / np.linalg.norm(good_feat)
        good_img = np.zeros((112, 112, 3), dtype=np.uint8)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [[100, 100, 200, 300, 1]]

        def mock_extract(det, rec, f, box, ratio, lock):
            return good_feat, good_img

        orig_extract = tracking.extract_feat
        tracking.extract_feat = mock_extract
        try:
            # Ngay sau khi block (chưa qua 0.3s):
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=10)
            self.assertEqual(len(tracker.gallery.snapshot()), 1, "Học phải bị chặn khi trong learn_block_sec")
            self.assertIsNone(tracker.learn_staging)

            # Chờ qua thời gian block
            time.sleep(0.35)

            # Mẫu thứ 1 sau khi hết block: đưa vào bộ đệm staging
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=11)
            self.assertEqual(len(tracker.gallery.snapshot()), 1)
            self.assertIsNotNone(tracker.learn_staging)

            # Mẫu thứ 2 sau khoảng learn_min_interval: xác nhận và commit vào gallery
            time.sleep(0.08)
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=12)
            self.assertEqual(len(tracker.gallery.snapshot()), 2, "Đã commit mẫu sau 2 lần xác nhận liên tiếp")
            self.assertIsNone(tracker.learn_staging)
        finally:
            tracking.extract_feat = orig_extract

    def test_07_two_sample_staging_buffer_rejection(self):
        """7. Bộ đệm chờ 2 mẫu: Nếu mẫu thứ 2 khác mẫu thứ 1 (pair_sim < 0.8), hủy bỏ mẫu nghi ngờ."""
        import tracking
        tracker = TrackingManager(MockDetector(face_box=True), MockRecognizer())
        tracker.learn_enabled = True
        tracker.learn_block_sec = 0.1
        tracker.learn_min_interval_sec = 0.05
        tracker.state = "TRACKING"
        tracker.locked_face_id = 1
        tracker.locked_bbox = [100, 100, 200, 300]
        tracker.target_bbox = [100, 100, 200, 300]

        anc = np.zeros(512, dtype=np.float32)
        anc[0] = 1.0
        tracker.gallery.add(anc, source="enroll")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [[100, 100, 200, 300, 1]]

        # Mẫu 1: sim ~ 0.85 với anchor
        feat1 = np.zeros(512, dtype=np.float32)
        feat1[0] = 0.85
        feat1[1] = 0.52678
        feat1 = feat1 / np.linalg.norm(feat1)

        # Mẫu 2: sim ~ 0.85 với anchor nhưng trục khác nhiều so với feat1 (pair_sim < 0.8)
        feat2 = np.zeros(512, dtype=np.float32)
        feat2[0] = 0.85
        feat2[2] = 0.52678
        feat2 = feat2 / np.linalg.norm(feat2)
        # pair_sim = feat1 . feat2 = 0.85 * 0.85 = 0.7225 < 0.80!

        current_feat = [feat1]

        def mock_extract(det, rec, f, box, ratio, lock):
            return current_feat[0], np.zeros((112, 112, 3), dtype=np.uint8)

        orig_extract = tracking.extract_feat
        tracking.extract_feat = mock_extract
        try:
            # Frame 1: nạp mẫu 1 vào staging
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=20)
            self.assertIsNotNone(tracker.learn_staging)
            self.assertEqual(len(tracker.gallery.snapshot()), 1)

            # Chờ interval
            time.sleep(0.08)

            # Frame 2: đưa mẫu 2 vào (pair_sim = 0.7225 < 0.80)
            current_feat[0] = feat2
            tracker.update(frame, detections, frame.shape[:2], det_frame=frame, det_fid=21)

            # Mẫu bị reject và staging được cập nhật bằng feat2 (không thêm vào gallery)
            self.assertEqual(len(tracker.gallery.snapshot()), 1, "Mẫu nghi ngờ không được thêm vào gallery")
        finally:
            tracking.extract_feat = orig_extract

    def test_08_person_id_tracker_gallery_helpers(self):
        """8. Kiểm tra các hàm trợ giúp của PersonIDTracker: snapshot, remove, set_anchor, clear."""
        pid_tracker = PersonIDTracker()
        tid = pid_tracker.next_id
        pid_tracker.tracked_faces[tid] = {
            "gallery": FaceGallery(gallery_max=5),
            "lost": 0,
            "t": time.monotonic()
        }

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        img1 = np.zeros((96, 96, 3), dtype=np.uint8)
        img2 = np.ones((96, 96, 3), dtype=np.uint8)

        pid_tracker.tracked_faces[tid]["gallery"].add(v1, img=img1, source="enroll")
        pid_tracker.tracked_faces[tid]["gallery"].add(v2, img=img2, source="learn")

        # Snapshot
        snap = pid_tracker.get_gallery_snapshot(tid)
        self.assertEqual(len(snap), 2)

        # Set anchor
        self.assertTrue(pid_tracker.set_gallery_anchor(tid, 1))
        snap2 = pid_tracker.get_gallery_snapshot(tid)
        self.assertTrue(np.allclose(snap2[0]["feat"], v2))

        # Remove item
        self.assertTrue(pid_tracker.remove_gallery_item(tid, 1))
        self.assertEqual(len(pid_tracker.get_gallery_snapshot(tid)), 1)

        # Clear gallery
        self.assertTrue(pid_tracker.clear_gallery(tid))
        self.assertEqual(len(pid_tracker.get_gallery_snapshot(tid)), 0)


if __name__ == "__main__":
    unittest.main()
