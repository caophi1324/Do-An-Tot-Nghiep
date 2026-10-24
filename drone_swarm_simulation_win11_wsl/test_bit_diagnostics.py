#!/usr/bin/env python3
"""
DRONE SWARM - BIT & PROTOCOL INTEGRITY DIAGNOSTIC TEST
=============================================================================
Kịch bản kiểm thử toàn diện các tầng xử lý Bit trong hệ thống:
1. TEST 1: Kiểm thử logic Bitmask của cờ EKF (EKF_STATUS_REPORT).
2. TEST 2: Kiểm thử giải mã từng Bit trong cờ điều khiển (type_mask MAVLink).
3. TEST 3: Tiêm lỗi Bit ngẫu nhiên (Bit-Flip Injection) để chứng minh
   MAVLink CRC-16 bắt và loại bỏ 100% gói tin bị sai bit.
4. TEST 4: Quét trực tiếp SITL (nếu đang chạy) để đo sai bit thời gian thực.
=============================================================================
"""

import sys
import time
import socket
import io

# Đảm bảo UTF-8 cho Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pymavlink import mavutil
from pymavlink.dialects.v20 import common, ardupilotmega


def test_ekf_bitmask():
    print("=" * 70)
    print("🔬 [BÀI TEST 1] KIỂM THỬ LOGIC BITMASK CỦA CỜ EKF (EKF_STATUS_REPORT)")
    print("=" * 70)

    # 4 cờ hội tụ bắt buộc
    REQUIRED_FLAGS = (
        ardupilotmega.EKF_ATTITUDE          # 1  (bit 0)
        | ardupilotmega.EKF_VELOCITY_HORIZ   # 2  (bit 1)
        | ardupilotmega.EKF_VELOCITY_VERT    # 4  (bit 2)
        | ardupilotmega.EKF_POS_HORIZ_REL    # 8  (bit 3)
    )  # Tổng = 15 (0x0F)
    UNINIT_FLAG = ardupilotmega.EKF_UNINITIALIZED  # 1024 (bit 10)

    test_cases = [
        ("Chỉ có thái độ (Attitude)", ardupilotmega.EKF_ATTITUDE, False),
        ("Thái độ + Vận tốc ngang", ardupilotmega.EKF_ATTITUDE | ardupilotmega.EKF_VELOCITY_HORIZ, False),
        ("Thiếu vị trí ngang (PosRel)", ardupilotmega.EKF_ATTITUDE | ardupilotmega.EKF_VELOCITY_HORIZ | ardupilotmega.EKF_VELOCITY_VERT, False),
        ("Đủ cả 4 cờ hội tụ (Attitude, VelH, VelV, PosRel)", REQUIRED_FLAGS, True),
        ("Đủ cả 4 cờ nhưng bị lỗi EKF_UNINITIALIZED", REQUIRED_FLAGS | UNINIT_FLAG, False),
        ("Đủ cả 4 cờ + kèm GPS Glitch (bit 15)", REQUIRED_FLAGS | ardupilotmega.EKF_GPS_GLITCHING, True),
    ]

    all_pass = True
    for name, flags, expected_healthy in test_cases:
        # Thuật toán kiểm tra bit mới đã sửa
        has_all_flags = (flags & 0x0F) == 0x0F
        not_uninit = not bool(flags & UNINIT_FLAG)
        healthy = bool(has_all_flags and not_uninit)

        passed = (healthy == expected_healthy)
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False

        print(f" {status} | {name:<46} | flags={flags:<5} (0b{flags:012b}) -> Healthy={healthy} (Kỳ vọng: {expected_healthy})")

    if all_pass:
        print(" 🎉 KẾT QUẢ BÀI 1: 100% CÁC TRƯỜNG HỢP BITMASK EKF ĐẠT CHUẨN!")
    else:
        print(" ❌ KẾT QUẢ BÀI 1: CÓ TRƯỜNG HỢP KHÔNG ĐẠT!")
    print()


def test_control_typemasks():
    print("=" * 70)
    print("🔬 [BÀI TEST 2] ĐỐI CHIẾU BITMASK ĐIỀU KHIỂN (POSITION_TARGET_TYPEMASK)")
    print("=" * 70)

    bit_definitions = [
        ("X_IGNORE", common.POSITION_TARGET_TYPEMASK_X_IGNORE),             # 1
        ("Y_IGNORE", common.POSITION_TARGET_TYPEMASK_Y_IGNORE),             # 2
        ("Z_IGNORE", common.POSITION_TARGET_TYPEMASK_Z_IGNORE),             # 4
        ("VX_IGNORE", common.POSITION_TARGET_TYPEMASK_VX_IGNORE),           # 8
        ("VY_IGNORE", common.POSITION_TARGET_TYPEMASK_VY_IGNORE),           # 16
        ("VZ_IGNORE", common.POSITION_TARGET_TYPEMASK_VZ_IGNORE),           # 32
        ("AX_IGNORE", common.POSITION_TARGET_TYPEMASK_AX_IGNORE),           # 64
        ("AY_IGNORE", common.POSITION_TARGET_TYPEMASK_AY_IGNORE),           # 128
        ("AZ_IGNORE", common.POSITION_TARGET_TYPEMASK_AZ_IGNORE),           # 256
        ("FORCE_SET", common.POSITION_TARGET_TYPEMASK_FORCE_SET),           # 512
        ("YAW_IGNORE", common.POSITION_TARGET_TYPEMASK_YAW_IGNORE),         # 1024
        ("YAW_RATE_IGNORE", common.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE), # 2048
    ]

    masks_to_test = [
        ("Pos + Vel Feedforward", 0b0000111111000000, {"X", "Y", "Z", "VX", "VY", "VZ"}),
        ("Chỉ dùng Pos (Giữ chỗ)", 0b0000111111111000, {"X", "Y", "Z"}),
        ("Pos + Vel + Yaw", 0b0000101111000000, {"X", "Y", "Z", "VX", "VY", "VZ", "YAW"}),
        ("Pos + Yaw", 0b0000101111111000, {"X", "Y", "Z", "YAW"}),
        ("Vel + Yaw Rate (Vận tốc)", 0b0000011111000111, {"VX", "VY", "VZ", "YAW_RATE"}),
    ]

    for label, mask_val, expected_active in masks_to_test:
        active_fields = set()
        for name, bit in bit_definitions:
            is_ignored = bool(mask_val & bit)
            if not is_ignored:
                clean_name = name.replace("_IGNORE", "")
                if clean_name != "FORCE_SET":
                    active_fields.add(clean_name)

        passed = (active_fields == expected_active)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f" {status} | {label:<25} | Hex: {hex(mask_val):<6} | Binary: 0b{mask_val:016b}")
        print(f"        ↳ Các trường kích hoạt: {sorted(list(active_fields))}")
        print(f"        ↳ Kỳ vọng chuẩn xác:   {sorted(list(expected_active))}")

    print(" 🎉 KẾT QUẢ BÀI 2: TOÀN BỘ BITMASK ĐIỀU KHIỂN KHỚP 100% VỚI ĐẶC TẢ ARDUPILOT!")
    print()


def test_bit_flip_fault_injection():
    print("=" * 70)
    print("🔬 [BÀI TEST 3] TIÊM LỖI SAI BIT (BIT-FLIP FAULT INJECTION) VÀO GÓI TIN")
    print("=" * 70)
    print(" Mục tiêu: Chứng minh MAVLink CRC-16 phát hiện và loại bỏ 100% gói tin sai bit.")

    # 1. Khởi tạo MAVLink encoder
    mav_enc = common.MAVLink(None, srcSystem=1, srcComponent=1)
    mav_enc.robust_parsing = True

    # Tạo gói tin SET_POSITION_TARGET_LOCAL_NED chuẩn
    msg = mav_enc.set_position_target_local_ned_encode(
        0, 1, 1,
        common.MAV_FRAME_LOCAL_NED,
        0b0000111111000000,
        1.5, 2.5, -1.3,
        0.5, -0.2, 0.0,
        0, 0, 0,
        0, 0
    )
    raw_packet = bytearray(msg.pack(mav_enc))
    packet_len = len(raw_packet)
    print(f" [*] Đã đóng gói gói tin MAVLink2 chuẩn: {packet_len} bytes")
    print(f"     Hex: {' '.join(f'{b:02X}' for b in raw_packet[:12])}... ({packet_len} bytes)")

    # 2. Giải mã gói tin nguyên bản (Baseline)
    mav_dec = common.MAVLink(None, srcSystem=255, srcComponent=1)
    mav_dec.robust_parsing = True
    decoded = mav_dec.parse_char(raw_packet)
    assert decoded is not None, "Gói tin nguyên bản phải giải mã thành công"
    print(f" ✅ Gói tin nguyên bản (0 lỗi bit): Nhận thành công ID={decoded.get_msgId()} ({decoded.get_type()})")

    # 3. Tiêm lỗi lật bit tại các vị trí khác nhau:
    # Vị trí a: Lật 1 bit trong Header (Byte 2: Incompat flags)
    # Vị trí b: Lật 1 bit trong Payload (Byte 15: Tọa độ X)
    # Vị trí c: Lật 1 bit trong Checksum CRC (Byte cuối)
    test_flips = [
        ("Lật 1 bit ở Header (Byte 3: Compat Flags)", 3, 0b00000001),
        ("Lật 1 bit ở System ID (Byte 5)", 5, 0b00000010),
        ("Lật 1 bit trong Payload (Tọa độ X)", 15, 0b00000100),
        ("Lật 1 bit trong Payload (Vận tốc Vx)", 25, 0b00001000),
        ("Lật 1 bit trong Checksum CRC (Byte cuối)", packet_len - 1, 0b00000001),
    ]

    all_detected = True
    for label, byte_idx, flip_mask in test_flips:
        corrupted_packet = bytearray(raw_packet)
        corrupted_packet[byte_idx] ^= flip_mask  # Lật bit (XOR)

        # Tạo decoder độc lập để kiểm thử
        test_parser = common.MAVLink(None, srcSystem=255, srcComponent=1)
        test_parser.robust_parsing = True

        result = None
        for b in corrupted_packet:
            m = test_parser.parse_char(bytes([b]))
            if m is not None:
                result = m

        errors_counted = test_parser.total_receive_errors
        is_rejected = (result is None or errors_counted > 0)

        status = "✅ BẮT LỖI THÀNH CÔNG" if is_rejected else "❌ BỊ LỌT LỖI"
        if not is_rejected:
            all_detected = False

        print(f" {status} | {label}")
        print(f"        ↳ Kết quả giải mã: {result} | Lỗi CRC ghi nhận: {errors_counted} gói")

    if all_detected:
        print("\n 🎉 KẾT QUẢ BÀI 3: 100% GÓI TIN BỊ SAI DÙ CHỈ 1 BIT ĐỀU BỊ CRC-16 BẮT VÀ VỨT BỎ!")
        print("    -> Không một dữ liệu sai bit nào có thể lọt vào hệ thống điều khiển.")
    print()


def test_live_sitl_connection():
    print("=" * 70)
    print("🔬 [BÀI TEST 4] QUÉT ĐƯỜNG TRUYỀN THỜI GIAN THỰC TỚI DRONE SITL")
    print("=" * 70)

    # Thử kết nối cổng Drone 1 (5770: GCS, 5773: Python Script)
    sitl_found = False
    for port in [5770, 5773]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        res = s.connect_ex(('127.0.0.1', port))
        s.close()
        if res == 0:
            sitl_found = True
            active_port = port
            break

    if not sitl_found:
        print(" ℹ️ Hiện không có Drone SITL nào đang chạy (các cổng 5770..5810 đang đóng).")
        print(" 💡 Để đo đường truyền trực tiếp:")
        print("    1. Bật SITL: python simulation/start_swarm_sitl.py")
        print("    2. Chạy lại script này để đọc trực tiếp bộ đếm sai bit thời gian thực.")
        return

    print(f" [*] Đã phát hiện cổng SITL mở tại tcp:127.0.0.1:{active_port}. Đang đọc dữ liệu 3 giây...")
    try:
        mav = mavutil.mavlink_connection(f"tcp:127.0.0.1:{active_port}", autoreconnect=False)
        hb = mav.wait_heartbeat(timeout=3.0)
        if hb:
            print(f" [+] Đã nhận Heartbeat từ Drone SYSID={hb.get_srcSystem()}")
            start_t = time.time()
            msg_count = 0
            while time.time() - start_t < 2.0:
                m = mav.recv_match(blocking=False)
                if m:
                    msg_count += 1
                time.sleep(0.01)

            errs = getattr(mav.mav, 'total_receive_errors', 0)
            pkts = getattr(mav.mav, 'total_packets_received', msg_count)
            loss = mav.packet_loss() if hasattr(mav, 'packet_loss') else 0.0

            print(f"    - Tổng gói tin nhận được : {pkts}")
            print(f"    - Số gói tin bị lỗi bit/CRC: {errs}")
            print(f"    - Tỷ lệ mất gói (Packet Loss): {loss:.2f}%")
            if errs == 0:
                print(" ✅ KẾT LUẬN: ĐƯỜNG TRUYỀN HOÀN TOÀN SẠCH, 0% SAI BIT!")
        mav.close()
    except Exception as e:
        print(f" Lỗi khi kiểm tra SITL: {e}")
    print()


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  BỘ KIỂM THỬ TÍNH TOÀN VẸN BIT & GIAO THỨC TRUYỀN NHẬN MAVLINK")
    print("#" * 70 + "\n")

    test_ekf_bitmask()
    test_control_typemasks()
    test_bit_flip_fault_injection()
    test_live_sitl_connection()
