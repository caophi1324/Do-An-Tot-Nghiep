import sys
import os
import unittest
import numpy as np

# Thêm đường dẫn pc_client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_client"))

from face_utils import iou, is_crowded, pick_face, blur_score, FaceGallery
from tracking import TrackingManager, BBoxOneEuroFilter
from app import PersonIDTracker, FaceDBProxy


class MockRecognizer:
    """Mock FaceRecognizerSF để kiểm thử matching sim."""
    def __init__(self, sim_map=None):
        self.sim_map = sim_map or {}

    def match(self, emb1, emb2, mode):
        # emb có thể là ndarray 1D chứa scalar nhận diện ID giả định
        val1 = float(emb1[0]) if hasattr(emb1, '__getitem__') else float(emb1)
        val2 = float(emb2[0]) if hasattr(emb2, '__getitem__') else float(emb2)
        key = (val1, val2)
        if key in self.sim_map:
            return self.sim_map[key]
        key_rev = (val2, val1)
        if key_rev in self.sim_map:
            return self.sim_map[key_rev]
        # Mặc định: nếu cùng giá trị thì sim = 0.9, khác giá trị thì sim = 0.1
        return 0.9 if abs(val1 - val2) < 1e-4 else 0.15


class MockDetector:
    """Mock FaceDetectorYN."""
    def __init__(self, faces=None):
        self.faces = faces

    def setInputSize(self, size):
        pass

    def detect(self, roi):
        return True, self.faces


class TestAntiIDSwap(unittest.TestCase):
    def test_01_iou_and_crowd_detection(self):
        # 1. Test IoU
        b1 = [0, 0, 100, 100]
        b2 = [0, 0, 100, 100]
        self.assertAlmostEqual(iou(b1, b2), 1.0, places=3)

        b3 = [200, 200, 300, 300]
        self.assertAlmostEqual(iou(b1, b3), 0.0, places=3)

        # 2. Test is_crowded
        boxA = [100, 100, 200, 300]
        boxB_far = [400, 100, 500, 300]
        self.assertFalse(is_crowded(boxA, [boxB_far]))

        boxB_overlapping = [150, 100, 250, 300]  # overlap > 10%
        self.assertTrue(is_crowded(boxA, [boxB_overlapping]))

    def test_02_face_gallery_anchor_retention(self):
        gallery = FaceGallery(maxlen=4)
        self.assertTrue(gallery.empty())

        emb_anchor = np.array([1.0] * 512, dtype=np.float32)
        gallery.add(emb_anchor)
        self.assertFalse(gallery.empty())
        self.assertTrue(np.allclose(gallery.anchor, emb_anchor))

        # Thêm các embedding tốt tiếp theo
        for i in range(5):
            gallery.add(np.array([float(i + 2)] * 512, dtype=np.float32))

        # Anchor gốc tuyệt đối KHÔNG bao giờ bị ghi đè!
        self.assertTrue(np.allclose(gallery.anchor, emb_anchor))
        self.assertEqual(len(gallery.items), 4)

        # Test best_sim
        rec = MockRecognizer()
        query_same = np.array([1.0] * 512, dtype=np.float32)
        sim = gallery.best_sim(rec, query_same)
        self.assertAlmostEqual(sim, 0.9, places=2)

    def test_03_facedb_proxy_compatibility(self):
        tracker = PersonIDTracker()
        tracker.next_id = 3
        # Giả lập track 1 có gallery với anchor, track 2 rỗng
        g1 = FaceGallery()
        g1.add(np.array([10.0] * 512, dtype=np.float32))
        tracker.tracked_faces[1] = {"bbox": (0, 0, 50, 100), "lost": 0, "t": 0.0, "vx": 0.0, "vy": 0.0, "gallery": g1}

        g2 = FaceGallery()
        tracker.tracked_faces[2] = {"bbox": (100, 0, 150, 100), "lost": 0, "t": 0.0, "vx": 0.0, "vy": 0.0, "gallery": g2}

        # Kiểm tra FaceDBProxy
        self.assertIn(1, tracker.face_db)
        self.assertNotIn(2, tracker.face_db)
        self.assertEqual(tracker.face_db.keys(), [1])
        self.assertIsNotNone(tracker.face_db.get(1))
        self.assertIsNone(tracker.face_db.get(2))

        # Kiểm tra xóa qua proxy
        del tracker.face_db[1]
        self.assertTrue(g1.empty())
        self.assertNotIn(1, tracker.face_db)

    def test_04_tracking_manager_properties_and_mismatch_drop(self):
        tm = TrackingManager(None, None)
        self.assertEqual(tm.state, "IDLE")

        # Test getter / setter cached_embedding tương thích ngược
        dummy_emb = np.array([5.0] * 512, dtype=np.float32)
        tm.cached_embedding = dummy_emb
        self.assertTrue(np.allclose(tm.cached_embedding, dummy_emb))
        self.assertFalse(tm.gallery.empty())

        tm.cached_embedding = None
        self.assertIsNone(tm.cached_embedding)
        self.assertTrue(tm.gallery.empty())

        # Test _drop_target
        tm.state = "TRACKING"
        tm.locked_bbox = [50, 50, 150, 250]
        tm.locked_face_id = 1
        tm._drop_target("Test mismatch drop")
        self.assertEqual(tm.state, "LOST")
        self.assertIsNone(tm.locked_bbox)
        self.assertIsNone(tm.locked_face_id)
        self.assertEqual(tm.mismatch, 0)

    def test_05_person_tracker_hungarian_assignment(self):
        tracker = PersonIDTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: 2 người xuất hiện tại 2 vị trí khác nhau
        dets_f1 = [
            [100, 100, 150, 300],  # Person A (bên trái)
            [400, 100, 450, 300],  # Person B (bên phải)
        ]
        res_f1 = tracker.update(dets_f1, frame, None)
        self.assertEqual(len(res_f1), 2)
        id_A = res_f1[0][4]
        id_B = res_f1[1][4]
        self.assertNotEqual(id_A, id_B)

        # Frame 2: Cả 2 người dịch chuyển một chút sang phải (+10 px)
        dets_f2 = [
            [110, 100, 160, 300],  # Person A moved
            [410, 100, 460, 300],  # Person B moved
        ]
        res_f2 = tracker.update(dets_f2, frame, None)
        assigned_A = [r[4] for r in res_f2 if abs(r[0] - 110) < 5][0]
        assigned_B = [r[4] for r in res_f2 if abs(r[0] - 410) < 5][0]
        # Hungarian phải ghép chính xác ID mà không bị lẫn
        self.assertEqual(assigned_A, id_A)
        self.assertEqual(assigned_B, id_B)

    def test_06_person_tracker_anti_id_swap_recovery(self):
        """
        Kiểm tra cơ chế tự động sửa swap:
        Nếu hai ID bị tráo nhầm, sau khi tách ra, kiểm tra face recognition
        phát hiện khuôn mặt lệch liên tiếp sẽ tự động sửa đảo lại ID chính xác.
        """
        tracker = PersonIDTracker()
        # Mock detector trả về 1 khuôn mặt giả định
        # [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y, score]
        mock_face = np.array([[10, 10, 50, 50, 20, 25, 40, 25, 30, 35, 25, 45, 35, 45, 0.95]], dtype=np.float32)
        tracker.detector = MockDetector(mock_face)

        # Mock recognizer:
        # ID 1 có embedding 1.0, ID 2 có embedding 2.0
        rec = MockRecognizer({
            (1.0, 1.0): 0.90,
            (2.0, 2.0): 0.90,
            (1.0, 2.0): 0.15,
        })
        tracker.recognizer = rec

        # Giả lập track 1 (gốc là Người 1, emb 1.0) và track 2 (gốc là Người 2, emb 2.0)
        g1 = FaceGallery()
        g1.add(np.array([1.0] * 512, dtype=np.float32))
        tracker.tracked_faces[1] = {"bbox": (100, 100, 150, 300), "lost": 0, "t": 0.0, "vx": 0.0, "vy": 0.0, "gallery": g1}

        g2 = FaceGallery()
        g2.add(np.array([2.0] * 512, dtype=np.float32))
        tracker.tracked_faces[2] = {"bbox": (400, 100, 450, 300), "lost": 0, "t": 0.0, "vx": 0.0, "vy": 0.0, "gallery": g2}

        # Giả lập tình huống assign bị swap nhầm do va chạm:
        # Box 0 (tọa độ người 1) bị gắn ID 2
        # Box 1 (tọa độ người 2) bị gắn ID 1
        assign = {0: 2, 1: 1}
        boxes = [(100, 100, 150, 300), (400, 100, 450, 300)]
        crowded = [False, False]

        # Thay extract_feat bằng lambda trả về embedding người 1 cho Box 0, người 2 cho Box 1
        import app
        import face_utils
        orig_extract = app.extract_feat
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_extract = lambda det, rec, fr, b, r, lk: (
            (np.array([1.0] * 512, dtype=np.float32), np.zeros((50, 50, 3), dtype=np.uint8))
            if b[0] < 200 else
            (np.array([2.0] * 512, dtype=np.float32), np.zeros((50, 50, 3), dtype=np.uint8))
        )
        try:
            app.extract_feat = mock_extract
            face_utils.extract_feat = mock_extract

            # Chạy _verify_and_swap qua 3 lần (settings swap_votes = 3)
            # Lưu ý c=0: (frame_idx + 0) % 5 == 0; c=1: (frame_idx + 1) % 5 == 0
            # Để cả hai box đều được kiểm tra mỗi lần lặp, ta chạy với các frame_idx phù hợp
            for step in range(3):
                # c=0 được check khi frame_idx % 5 == 0
                tracker.frame_idx = step * 10
                tracker._verify_and_swap(assign, boxes, crowded, dummy_frame, None, 0.45, 0.363)
                # c=1 được check khi (frame_idx + 1) % 5 == 0 -> frame_idx = 4 + 10 * step
                tracker.frame_idx = step * 10 + 4
                tracker._verify_and_swap(assign, boxes, crowded, dummy_frame, None, 0.45, 0.363)

            # Sau 3 lần vote, assign PHẢI được tự động đảo lại đúng:
            # Box 0 -> ID 1, Box 1 -> ID 2
            self.assertEqual(assign[0], 1)
            self.assertEqual(assign[1], 2)
        finally:
            app.extract_feat = orig_extract
            face_utils.extract_feat = orig_extract

    def test_07_scenario_opposite_crossing_motion_prediction(self):
        """
        Kịch bản 1: Hai người đi ngược chiều, cắt nhau.
        Nhờ dự đoán vị trí bằng vận tốc, 2 track tiếp tục đi theo quán tính qua điểm giao nhau.
        """
        tracker = PersonIDTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Person 1 di chuyển từ trái sang phải: x=100 -> 140 -> 180
        # Person 2 di chuyển từ phải sang trái: x=400 -> 360 -> 320
        # Mô phỏng chu kỳ frame thực tế với dt ~ 0.05s (20 FPS)
        import time
        d1 = [[100, 100, 140, 300], [400, 100, 440, 300]]
        res1 = tracker.update(d1, frame, None)
        id1 = [r[4] for r in res1 if r[0] < 200][0]
        id2 = [r[4] for r in res1 if r[0] > 300][0]

        time.sleep(0.05)
        d2 = [[140, 100, 180, 300], [360, 100, 400, 300]]
        res2 = tracker.update(d2, frame, None)

        time.sleep(0.05)
        d3 = [[180, 100, 220, 300], [320, 100, 360, 300]]
        res3 = tracker.update(d3, frame, None)

        # Điểm giao nhau ở giữa: x=240 và x=260 (crowded)
        time.sleep(0.05)
        d_cross = [[240, 100, 280, 300], [250, 100, 290, 300]]
        res_cross = tracker.update(d_cross, frame, None)
        self.assertEqual(len(res_cross), 2)

        # Sau khi cắt qua nhau:
        # Person 1 tiếp tục sang phải: x=300
        # Person 2 tiếp tục sang trái: x=200
        time.sleep(0.05)
        d_after = [[200, 100, 240, 300], [300, 100, 340, 300]]
        res_after = tracker.update(d_after, frame, None)
        id_left = [r[4] for r in res_after if r[0] == 200][0]
        id_right = [r[4] for r in res_after if r[0] == 300][0]

        # ID của người đi từ phải sang trái (id2) giờ phải ở bên trái (x=200)
        # ID của người đi từ trái sang phải (id1) giờ phải ở bên phải (x=300)
        self.assertEqual(id_left, id2)
        self.assertEqual(id_right, id1)

    def test_08_scenario_relock_margin_and_votes(self):
        """
        Kịch bản 5: Smart relock bằng SFace cần thỏa:
        1. margin_ok: chênh lệch điểm với ứng viên thứ 2 >= 0.05
        2. Cần ít nhất 2 vote liên tiếp mới relock để tránh frame may mắn
        """
        from config import settings
        rec_interval = settings.recognition.get("recognition_interval_frames", 3)

        tm = TrackingManager(MockDetector(), None)
        tm.state = "LOST"
        # Anchor là ID 1
        g = FaceGallery()
        g.add(np.array([1.0] * 512, dtype=np.float32))
        tm.gallery = g
        tm.recognizer = MockRecognizer({
            (1.0, 1.0): 0.85,  # Ứng viên 1: sim 0.85
            (1.0, 2.0): 0.82,  # Ứng viên 2: sim 0.82 (margin = 0.03 < 0.05 -> không an toàn!)
        })

        import face_utils
        import tracking
        orig_extract_fu = face_utils.extract_feat
        orig_extract_tm = tracking.extract_feat
        mock_extract = lambda det, rec, fr, b, r, lk: (
            (np.array([1.0] * 512, dtype=np.float32), np.zeros((10, 10))) if b[4] == 10 else
            (np.array([2.0] * 512, dtype=np.float32), np.zeros((10, 10)))
        )
        try:
            face_utils.extract_feat = mock_extract
            tracking.extract_feat = mock_extract

            detections = [
                [100, 100, 200, 300, 10],
                [400, 100, 500, 300, 20],
            ]
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame trigger scan nhưng margin < 0.05 -> KHÔNG relock
            tm.lost_frame_counter = rec_interval - 1
            res = tm.update(frame, detections, (480, 640))
            self.assertEqual(res["state"], "LOST")

            # Bây giờ ứng viên 2 có sim thấp 0.3 (margin = 0.85 - 0.30 = 0.55 >= 0.05)
            tm.recognizer.sim_map[(1.0, 2.0)] = 0.30

            # Lần quét 1: vote lần 1 cho ID 10
            tm.lost_frame_counter = rec_interval - 1
            res = tm.update(frame, detections, (480, 640))
            self.assertEqual(res["state"], "LOST")  # Chưa đủ 2 vote, vẫn LOST
            self.assertEqual(tm.relock_votes.get(10), 1)

            # Lần quét 2: vote lần 2 cho ID 10 -> RELOCK THÀNH CÔNG!
            tm.lost_frame_counter = rec_interval - 1
            res = tm.update(frame, detections, (480, 640))
            self.assertEqual(res["state"], "TRACKING")
            self.assertEqual(tm.locked_face_id, 10)
        finally:
            face_utils.extract_feat = orig_extract_fu
            tracking.extract_feat = orig_extract_tm

    def test_09_scenario_third_person_rejection(self):
        """
        Kịch bản 6: Khi đang khóa ID 1, sự xuất hiện của người thứ 3 không làm nhảy mục tiêu.
        """
        tm = TrackingManager(None, None)
        tm.state = "TRACKING"
        tm.locked_bbox = [100, 100, 200, 300]
        tm._raw_locked_bbox = [100, 100, 200, 300]
        tm.locked_face_id = 1

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Chỉ có người thứ 3 (ID 3) đứng rất gần vị trí cũ
        detections = [
            [105, 105, 205, 305, 3],  # Người 3
        ]
        res = tm.update(frame, detections, (480, 640))
        # TrackingManager phải từ chối ID 3 và chuyển sang Grace Period hoặc LOST thay vì nhảy sang ID 3
        self.assertNotEqual(tm.locked_face_id, 3)
        self.assertEqual(tm.locked_face_id, 1)

    def test_10_tighten_bbox_face_interval_throttling(self):
        """
        Đợt 2 - Mục 6: tighten_bbox chỉ chạy detector mỗi N frame (tighten_face_interval),
        các frame còn lại tái sử dụng độ lệch đỉnh đầu _last_top_offset.
        """
        class CountingDetector:
            def __init__(self):
                self.call_count = 0
            def setInputSize(self, size):
                pass
            def detect(self, roi):
                self.call_count += 1
                # Trả về 1 mặt ở y=10, h=30 -> abs_fy1 = 10 + 10 = 20, margin = 4.5
                return True, np.array([[10, 10, 30, 30]], dtype=np.float32)

        detector = CountingDetector()
        tm = TrackingManager(detector, None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [100.0, 100.0, 200.0, 300.0]

        from config import settings
        settings.set("tracking", "tighten_face_interval", 5)

        # Chạy 10 frame
        for f in range(10):
            res_box = tm.tighten_bbox(bbox, frame, 640, 480)
            self.assertIsNotNone(res_box)

        # 10 frame được gọi detect đúng 3 lần (frame 1 do _last_top_offset is None, frame 5, và frame 10)
        self.assertEqual(detector.call_count, 3)
        self.assertIsNotNone(tm._last_top_offset)

    def test_11_anti_gallery_contamination_and_config_sync(self):
        """
        Đợt 1 - Mục 2, 3 & Đợt 2 - Mục 7:
        - Đồng bộ 2 chiều giữa face_threshold và sface_threshold
        - det_frame được truyền an toàn vào tm.update
        - Chống nhiễm gallery: không add khi độ chênh lệch sim với người khác < gallery_add_margin
        """
        from config import settings
        # 1. Test đồng bộ cài đặt 2 chiều
        settings.set("recognition", "face_threshold", 0.42)
        self.assertAlmostEqual(settings.recognition["sface_threshold"], 0.42, places=3)
        settings.set("recognition", "sface_threshold", 0.38)
        self.assertAlmostEqual(settings.recognition["face_threshold"], 0.38, places=3)

        # Khôi phục mặc định
        settings.set("recognition", "face_threshold", 0.363)

        # 2. Test anti-gallery contamination trong TrackingManager.update
        tm = TrackingManager(MockDetector(), None)
        tm.state = "TRACKING"
        tm.locked_face_id = 1
        tm.locked_bbox = [100, 100, 200, 300]
        tm._raw_locked_bbox = [100, 100, 200, 300]

        g = FaceGallery()
        emb_anchor = np.array([1.0] * 512, dtype=np.float32)
        g.add(emb_anchor)
        tm.gallery = g

        # Recognizer: Target sim = 0.80, Other person sim = 0.75
        # Margin = 0.80 - 0.75 = 0.05 < gallery_add_margin (0.10) -> KHÔNG ĐƯỢC ADD!
        rec = MockRecognizer({
            (1.0, 1.0): 0.80,
            (1.0, 2.0): 0.75,
        })
        tm.recognizer = rec

        import face_utils
        import tracking
        orig_extract_fu = face_utils.extract_feat
        orig_extract_tm = tracking.extract_feat

        mock_extract = lambda det, r, fr, b, ratio, lk: (
            (np.array([2.0] * 512, dtype=np.float32), np.zeros((10, 10))) if ((len(b) >= 5 and b[4] == 2) or b[0] > 300) else
            (np.array([1.0] * 512, dtype=np.float32), np.zeros((10, 10)))
        )
        try:
            face_utils.extract_feat = mock_extract
            tracking.extract_feat = mock_extract

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            det_frame = np.ones((480, 640, 3), dtype=np.uint8)
            detections = [
                [100, 100, 200, 300, 1],  # Locked target
                [350, 100, 450, 300, 2],  # Other person nearby
            ]

            initial_gallery_count = len(tm.gallery.items)
            # rec_interval = 10, đặt counter trước 1 frame để trigger
            rec_interval = settings.recognition.get("recognition_interval_frames", 10)
            tm.sim_frame_counter = rec_interval - 1

            res = tm.update(frame, detections, (480, 640), det_frame=det_frame)
            self.assertEqual(res["state"], "TRACKING")
            # Do margin_ok = False (0.80 - 0.75 = 0.05 < 0.10), gallery KHÔNG được add thêm
            self.assertEqual(len(tm.gallery.items), initial_gallery_count)

            # Trường hợp an toàn: Other person sim = 0.20 (Margin = 0.80 - 0.20 = 0.60 >= 0.10)
            rec.sim_map[(1.0, 2.0)] = 0.20
            tm.sim_frame_counter = rec_interval - 1
            res2 = tm.update(frame, detections, (480, 640), det_frame=det_frame)
            self.assertEqual(res2["state"], "TRACKING")
            # Xác nhận mẫu thứ 2 liên tiếp qua bộ đệm chờ (chống nhiễm)
            tm.sim_frame_counter = rec_interval - 1
            res3 = tm.update(frame, detections, (480, 640), det_frame=det_frame)
            self.assertEqual(res3["state"], "TRACKING")
            # Bây giờ đã an toàn -> gallery được add thêm 1 feature mới
            self.assertEqual(len(tm.gallery.items), initial_gallery_count + 1)
        finally:
            face_utils.extract_feat = orig_extract_fu
            tracking.extract_feat = orig_extract_tm


if __name__ == "__main__":
    unittest.main()
