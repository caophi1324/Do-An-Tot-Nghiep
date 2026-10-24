import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pc_client'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import unittest
import numpy as np
import tracking as T
from config import settings

FW, FH = 640, 480
def unit(v): return v / np.linalg.norm(v)

class Frame:
    shape = (FH, FW, 3)

rng = np.random.default_rng(0)
A = unit(rng.normal(size=128))
B = unit(rng.normal(size=128))
BOX = (200, 100, 300, 360, 1)      # x1, y1, x2, y2, id


class TestTrackingManager(unittest.TestCase):
    def setUp(self):
        settings.recognition.update({"recognition_interval_frames": 10, "mismatch_limit": 3, "profile_enabled": False})
        self.state = {"feat": None, "calls": 0}
        self.orig_extract_feat = T.extract_feat
        def fake_extract(det, rec, frame, bbox, hr, lock=None):
            self.state["calls"] += 1
            return (self.state["feat"], None) if self.state["feat"] is not None else (None, None)
        T.extract_feat = fake_extract
        class Rec:
            def match(self, a, b, d=None):
                a, b = np.ravel(a), np.ravel(b)
                return float(a @ b)
        self.tm = T.TrackingManager(detector=object(), recognizer=Rec())
        self.tm.state_ = self.state

    def tearDown(self):
        T.extract_feat = self.orig_extract_feat

    def lock(self, feat):
        self.tm.state_["feat"] = feat
        self.tm.lock_target_by_bbox(Frame(), BOX, FW, FH)

    def test_cung_mot_frame_yolo_khong_kiem_tra_lap(self):
        self.lock(A)
        self.tm.state_["calls"] = 0
        for _ in range(100):                               # 100 tick nhưng YOLO chỉ ra 1 kết quả (fid=7)
            self.tm.update(Frame(), [BOX], (FH, FW), det_fid=7)
        self.assertLessEqual(self.tm.state_["calls"], 1, f"Expected <= 1, got {self.tm.state_['calls']}")

    def test_khong_drop_chi_vi_mot_quan_sat_xau_lap_lai(self):
        self.lock(A)
        self.tm.state_["feat"] = B                              # quan sát sai người (hoặc mặt nhiễu)
        for _ in range(200):
            r = self.tm.update(Frame(), [BOX], (FH, FW), det_fid=7)   # nhưng chỉ là 1 frame YOLO, lặp lại
        self.assertEqual(r["state"], "TRACKING")

    def test_van_drop_khi_that_su_lech_nhieu_frame(self):
        self.lock(A)
        self.tm.state_["feat"] = B
        state = None
        for fid in range(1, 200):                          # mỗi tick là 1 frame YOLO mới
            state = self.tm.update(Frame(), [BOX], (FH, FW), det_fid=fid)["state"]
            if state == "LOST":
                break
        self.assertEqual(state, "LOST")

    def test_tuong_thich_nguoc_khi_khong_truyen_det_fid(self):
        self.lock(A)
        self.tm.state_["calls"] = 0
        for _ in range(50):
            self.tm.update(Frame(), [BOX], (FH, FW))            # gọi kiểu cũ
        self.assertEqual(self.tm.state_["calls"], 5)            # 50/10, như trước

    def test_wait_for_id_don_sach_profile_cu(self):
        self.lock(A)
        self.tm.profile, self.tm.profile_status, self.tm.mismatch = object(), "ready", 2
        self.tm.wait_for_id(9, history=[B, A])
        self.assertIsNone(self.tm.profile)
        self.assertEqual(self.tm.profile_status, "none")
        self.assertEqual(self.tm.mismatch, 0)
        self.assertEqual(self.tm.locked_face_id, 9)
        self.assertEqual(self.tm.state, "LOST")
        self.assertIsNotNone(self.tm.gallery.anchor)
        self.assertEqual(len(self.tm.gallery.items), 1)

    def test_wait_for_id_roi_relock_khi_id_xuat_hien(self):
        self.lock(A)
        self.tm.wait_for_id(1, history=[A])
        r = self.tm.update(Frame(), [BOX], (FH, FW), det_fid=1)
        self.assertEqual(r["state"], "TRACKING")


if __name__ == "__main__":
    unittest.main()
