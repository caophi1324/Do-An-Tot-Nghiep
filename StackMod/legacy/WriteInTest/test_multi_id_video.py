#!/usr/bin/env python3
"""
Multi-Person AI Tracking & ID Benchmark on Video
================================================
Kiểm thử nhận diện nhiều người và gán ID (PersonIDTracker + YOLOv11 + SFace) trên video.
- Chạy: python test_multi_id_video.py [duong_dan_video]
- Phím tắt:
    [Click chuột trái] : Khóa bám đuổi người được chọn
    [A]                : Tự động khóa người gần tâm nhất
    [R]                : Hủy khóa (Reset lock)
    [SPACE]            : Tạm dừng / Tiếp tục video
    [S]                : Nhích từng frame (khi đang pause)
    [Q] / [ESC]        : Thoát
"""

import sys
import os
import time
import cv2
import numpy as np

# Thiết lập đường dẫn thư mục pc_client
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PC_CLIENT_DIR = os.path.join(BASE_DIR, "pc_client")
sys.path.insert(0, PC_CLIENT_DIR)

from config import settings
from app import PersonIDTracker
from tracking import TrackingManager

# Bảng màu sắc trực quan phân biệt từng ID người
PALETTE = [
    (0, 255, 255),    # ID 1: Vàng Cyan
    (0, 255, 0),      # ID 2: Xanh lá Neon
    (0, 165, 255),    # ID 3: Cam sáng
    (255, 0, 255),    # ID 4: Tím hồng Magenta
    (255, 255, 0),    # ID 5: Vàng Chanh
    (255, 140, 0),    # ID 6: Xanh biển sâu
    (147, 20, 255),   # ID 7: Hồng đậm
    (0, 215, 255),    # ID 8: Vàng kim
    (128, 255, 0),    # ID 9: Xanh nõn chuối
    (203, 192, 255),  # ID 10: Hồng phấn
]

def get_color(track_id):
    if track_id is None:
        return (200, 200, 200)
    return PALETTE[(int(track_id) - 1) % len(PALETTE)]

def main():
    video_source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "youtube_test.mp4")
    if not os.path.exists(video_source):
        cand = os.path.join(BASE_DIR, os.path.basename(video_source))
        if os.path.exists(cand):
            video_source = cand
        elif os.path.exists(os.path.join(BASE_DIR, "test_pedestrians.avi")):
            video_source = os.path.join(BASE_DIR, "test_pedestrians.avi")
    if not os.path.exists(video_source):
        print(f"[ERROR] Không tìm thấy file video: {video_source}")
        return

    print("=" * 70)
    print(f"🎬 BẮT ĐẦU TEST MULTI-PERSON TRACKING & ID TRÊN VIDEO")
    print(f"📁 Video: {video_source}")
    print("=" * 70)
    print("🎮 Phím tắt & Thao tác điều khiển:")
    print("   • Click chuột trái vào người để KHÓA bám đuổi ID đó")
    print("   • [A] : Tự động khóa người gần tâm nhất")
    print("   • [R] : Hủy khóa mục tiêu (Reset target)")
    print("   • [C] : XÓA TOÀN BỘ CSDL ID & KHUÔN MẶT (Reset ID đếm lại từ 1)")
    print("   • [SPACE] : Tạm dừng / Tiếp tục")
    print("   • [S] : Xem từng frame (khi pause)")
    print("   • [F] / [B] : Tua tới / Tua lùi 3s")
    print("   • [J] : Nhảy tới đoạn nhảy sân khấu (frame 3800)")
    print("   • [Q] / [ESC] : Thoát")
    print("   • Click chuột trực tiếp vào nút [Huy Khoa (R)] hoặc [XOA CSDL (C)] góc dưới")
    print("=" * 70)

    # 1. Nạp mô hình AI
    print("[1/3] Nạp mô hình YOLOv11...")
    from ultralytics import YOLO
    yolo_model = YOLO(settings.recognition.get("yolo_path"))

    backend = settings.recognition.get("recognizer_backend", "sface")
    print(f"[2/3] Nạp mô hình YuNet & Recognizer (Backend: {backend})...")
    yunet_path = settings.recognition.get("yunet_path")
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320)) if os.path.exists(yunet_path) else None
    
    if backend == "arcface":
        from arcface_recognizer import ArcFaceRecognizer
        rec_path = settings.recognition.get("arcface_path", os.path.join(PC_CLIENT_DIR, "w600k_r50.onnx"))
        recognizer = ArcFaceRecognizer(rec_path, device_id=settings.recognition.get("dml_device_id", 0)) if os.path.exists(rec_path) else None
    else:
        sface_path = settings.recognition.get("sface_path")
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "") if os.path.exists(sface_path) else None

    print("[3/3] Khởi tạo PersonIDTracker & TrackingManager...")
    face_tracker = PersonIDTracker(detector, recognizer)
    tracking_mgr = TrackingManager(detector, recognizer, face_tracker=face_tracker)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {video_source}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = max(1, int(1000.0 / video_fps))

    win_name = "AI Multi-Person Tracking & ID Benchmark"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 768)

    # Thanh trượt tua video (Trackbar)
    is_seeking = False
    updating_trackbar_ui = False

    def on_trackbar(val):
        nonlocal frame_idx, is_seeking
        if updating_trackbar_ui:
            return  # Không seek nếu giá trị do vòng lặp phát video cập nhật
        if not is_seeking:
            is_seeking = True
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            frame_idx = val
            is_seeking = False

    if total_frames > 0:
        cv2.createTrackbar("Thoi Gian", win_name, 0, total_frames, on_trackbar)

    # Biến tương tác chuột và nút bấm giao diện
    clicked_pos = None
    btn_action = None
    btn_reset_rect = (0, 0, 0, 0)
    btn_cleardb_rect = (0, 0, 0, 0)
    notice_text = ""
    notice_time = 0.0

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_pos, btn_action
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_reset_rect[0] <= x <= btn_reset_rect[2] and btn_reset_rect[1] <= y <= btn_reset_rect[3]:
                btn_action = "reset"
            elif btn_cleardb_rect[0] <= x <= btn_cleardb_rect[2] and btn_cleardb_rect[1] <= y <= btn_cleardb_rect[3]:
                btn_action = "clear_db"
            else:
                clicked_pos = (x, y)

    cv2.setMouseCallback(win_name, on_mouse)

    paused = False
    frame_idx = 0
    t_last = time.time()
    fps_smooth = video_fps

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Tua lại đầu video để lặp lại
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    break
            frame_idx += 1
            if total_frames > 0 and frame_idx % 30 == 0:
                updating_trackbar_ui = True
                cv2.setTrackbarPos("Thoi Gian", win_name, min(total_frames, frame_idx))
                updating_trackbar_ui = False
        else:
            time.sleep(0.02)

        fh, fw = frame.shape[:2]
        display_frame = frame.copy()
        btn_reset_rect = (fw - 275, fh - 28, fw - 150, fh - 4)
        btn_cleardb_rect = (fw - 145, fh - 28, fw - 10, fh - 4)

        # Xử lý hành động từ nút bấm giao diện
        if btn_action == "reset":
            tracking_mgr.reset()
            notice_text = "DA HUY KHOA MUC TIEU"
            notice_time = time.time() + 2.0
            print("\n[RESET] Đã hủy khóa mục tiêu.")
            btn_action = None
        elif btn_action == "clear_db":
            tracking_mgr.reset()
            face_tracker.clear()
            notice_text = "DA XOA TOAN BO CSDL - RESET ID VE 1!"
            notice_time = time.time() + 2.5
            print("\n[RESET-DB] Đã xóa toàn bộ CSDL khuôn mặt và ID (bộ đếm ID bắt đầu lại từ 1).")
            btn_action = None

        # 1. Chạy YOLO phát hiện người (Class 0: Person)
        t0 = time.monotonic()
        results = yolo_model(frame, verbose=False)
        yolo_ms = (time.monotonic() - t0) * 1000

        dets = []
        conf_thres = settings.recognition.get("conf_threshold", 0.4)
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= conf_thres and int(box.cls[0]) == 0:
                    dets.append(box.xyxy[0].tolist())

        # 2. Gán ID đa đối tượng bằng PersonIDTracker (Hungarian + Velocity + SFace)
        locked_id = tracking_mgr.locked_face_id
        dets_with_id = face_tracker.update(dets, frame, None, locked_face_id=locked_id)

        # 3. Xử lý tương tác click-to-lock
        if clicked_pos is not None:
            cx, cy = clicked_pos
            best_det = None
            min_d = 999999
            for b in dets_with_id:
                bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                d = (bx - cx)**2 + (by - cy)**2
                if d < min_d and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    min_d = d
                    best_det = b
            if best_det is not None:
                hist = face_tracker.get_history_with_images(best_det[4]) if len(best_det) >= 5 else []
                tracking_mgr.lock_target_by_bbox(frame, best_det, fw, fh, history=hist)
                print(f"\n[CLICK-LOCK] Đã chọn khóa mục tiêu ID: {best_det[4]}")
            else:
                tracking_mgr.reset()
                print("\n[CLICK-UNLOCK] Đã hủy khóa mục tiêu.")
            clicked_pos = None

        # 4. Cập nhật TrackingManager
        track_res = tracking_mgr.update(frame, dets_with_id, (fh, fw), det_frame=frame)
        track_state = track_res.get("state", "IDLE")
        locked_bbox = track_res.get("locked_bbox")

        # Đo FPS hiển thị
        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(0.001, now - t_last))
        t_last = now

        # 5. Vẽ Bounding Box của tất cả người xuất hiện
        for det in dets_with_id:
            x1, y1, x2, y2, tid = [int(v) for v in det[:5]]
            is_locked = (tracking_mgr.locked_face_id == tid and track_state == "TRACKING")
            color = (0, 215, 255) if is_locked else get_color(tid)
            thickness = 3 if is_locked else 2

            # Bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Nhãn ID
            label = f"★ LOCKED ID: {tid}" if is_locked else f"ID: {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 4, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # 6. Vẽ Bounding Box đã được lọc làm mượt (1 Euro Filter & Tighten) của mục tiêu đang khóa
        if is_locked and locked_bbox is not None:
            lx1, ly1, lx2, ly2 = [int(v) for v in locked_bbox[:4]]
            cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            # Vẽ tâm ngắm
            tcx, tcy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
            cv2.drawMarker(display_frame, (tcx, tcy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # 7. Vẽ Thanh Thông Tin Trên Cùng (HUD Banner)
        cv2.rectangle(display_frame, (0, 0), (fw, 45), (25, 25, 25), -1)
        cv2.line(display_frame, (0, 45), (fw, 45), (60, 60, 60), 1)

        status_color = (0, 255, 0) if track_state == "TRACKING" else (0, 255, 255) if track_state == "LOST" else (180, 180, 180)
        prof_info = track_res.get("profile", {})
        p_status = prof_info.get("status", "none")
        prof_str = ""
        if p_status == "learning":
            pn, pN = prof_info.get("progress", (0, 0))
            prof_str = f" | Profile: learning {pn}/{pN}"
        elif p_status == "ready":
            pstats = prof_info.get("stats", {})
            prof_str = f" | Profile: ready ({pstats.get('fixed',0)} fix, {pstats.get('adaptive',0)} adp)"
        elif p_status.startswith("failed:"):
            prof_str = f" | Profile: {p_status}"
        dec_source = track_res.get("decision_source", "hình học")
        hud_text = f"STATE: {track_state} | TARGET ID: {tracking_mgr.locked_face_id or 'None'} ({dec_source}){prof_str}"
        cv2.putText(display_frame, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        rec_tag = "ArcFace (DirectML GPU)" if backend == "arcface" else "SFace (CPU)"
        perf_text = f"MODEL: {rec_tag} | YOLO: {yolo_ms:.1f}ms | FPS: {fps_smooth:.1f} | Frame: {frame_idx}/{total_frames}"
        (pw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(display_frame, perf_text, (fw - pw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Hướng dẫn thao tác và các nút bấm dưới đáy
        cv2.rectangle(display_frame, (0, fh - 32), (fw, fh), (20, 20, 20), -1)
        cv2.line(display_frame, (0, fh - 32), (fw, fh - 32), (50, 50, 50), 1)

        info_text = "[Click]: Khoa | [A]: Auto | [R]: Huy Khoa | [C]: Xoa CSDL | [SPACE]: Pause | [F/B]: Tua | [J]: F=3800"
        cv2.putText(display_frame, info_text, (15, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # Nút Hủy Khóa (R)
        cv2.rectangle(display_frame, (btn_reset_rect[0], btn_reset_rect[1]), (btn_reset_rect[2], btn_reset_rect[3]), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (btn_reset_rect[0], btn_reset_rect[1]), (btn_reset_rect[2], btn_reset_rect[3]), (180, 180, 180), 1)
        cv2.putText(display_frame, "Huy Khoa (R)", (btn_reset_rect[0] + 12, btn_reset_rect[1] + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

        # Nút Xóa CSDL (C)
        cv2.rectangle(display_frame, (btn_cleardb_rect[0], btn_cleardb_rect[1]), (btn_cleardb_rect[2], btn_cleardb_rect[3]), (35, 35, 170), -1)
        cv2.rectangle(display_frame, (btn_cleardb_rect[0], btn_cleardb_rect[1]), (btn_cleardb_rect[2], btn_cleardb_rect[3]), (90, 90, 255), 1)
        cv2.putText(display_frame, "XOA CSDL (C)", (btn_cleardb_rect[0] + 12, btn_cleardb_rect[1] + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Thông báo nổi trên màn hình khi thực hiện thao tác
        if time.time() < notice_time and notice_text:
            (nw, nh), _ = cv2.getTextSize(notice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ncx = (fw - nw) // 2
            ncy = 80
            cv2.rectangle(display_frame, (ncx - 15, ncy - nh - 10), (ncx + nw + 15, ncy + 10), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (ncx - 15, ncy - nh - 10), (ncx + nw + 15, ncy + 10), (0, 215, 255), 2)
            cv2.putText(display_frame, notice_text, (ncx, ncy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        elif key in (ord(' '), ):  # Space
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}]")
        elif key in (ord('s'), ord('S')) and paused:
            # Nhích 1 frame
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_idx += 1
        elif key in (ord('f'), ord('F'), ord('.'), ord('>')):
            # Tua tới 150 frames (~3 giây)
            target_f = min(total_frames - 1, frame_idx + 150)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            frame_idx = target_f
            print(f"\n[SEEK] >> Tua tới frame {frame_idx}/{total_frames}")
        elif key in (ord('b'), ord('B'), ord(','), ord('<')):
            # Tua lùi 150 frames (~3 giây)
            target_f = max(0, frame_idx - 150)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            frame_idx = target_f
            print(f"\n[SEEK] << Tua lùi frame {frame_idx}/{total_frames}")
        elif key in (ord('j'), ord('J')):
            # Nhảy thẳng tới đoạn nhảy sân khấu (frame 3800)
            target_f = min(total_frames - 1, 3800)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            frame_idx = target_f
            print(f"\n[JUMP] >>> Nhảy thẳng tới đoạn bài nhảy 3 người (frame {frame_idx})")
        elif key in (ord('r'), ord('R')):
            tracking_mgr.reset()
            notice_text = "DA HUY KHOA MUC TIEU"
            notice_time = time.time() + 2.0
            print("\n[RESET] Đã hủy khóa mục tiêu.")
        elif key in (ord('c'), ord('C')):
            tracking_mgr.reset()
            face_tracker.clear()
            notice_text = "DA XOA TOAN BO CSDL - RESET ID VE 1!"
            notice_time = time.time() + 2.5
            print("\n[RESET-DB] Đã xóa toàn bộ CSDL khuôn mặt và ID (bộ đếm ID bắt đầu lại từ 1).")
        elif key in (ord('a'), ord('A')):
            # Auto-lock người gần tâm nhất
            if dets_with_id:
                cx_center, cy_center = fw / 2, fh / 2
                best_det = min(dets_with_id, key=lambda b: ((b[0]+b[2])/2 - cx_center)**2 + ((b[1]+b[3])/2 - cy_center)**2)
                hist = face_tracker.get_history_with_images(best_det[4]) if len(best_det) >= 5 else []
                tracking_mgr.lock_target_by_bbox(frame, best_det, fw, fh, history=hist)
                print(f"\n[AUTO-LOCK] Đã tự động khóa người gần tâm: ID {best_det[4]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Đã kết thúc phiên test video.")

if __name__ == "__main__":
    main()
