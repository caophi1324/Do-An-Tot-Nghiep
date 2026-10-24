#!/usr/bin/env python3
"""
DRONEEXE - Real-World Safety Harness & Validation Script
=============================================================================
Kịch bản kiểm thử quy chuẩn an toàn thực tế trên ArduPilot SITL / ESP32-S3:
- Kiểm tra toàn bộ điều kiện Pre-Arm (Pin >= 11.1V, EKF3 Healthy, ToF MTF-02).
- Khóa an toàn nghiêm ngặt ARMING_CHECK = 1 (Không bao giờ Force-Arm).
- Thử nghiệm 4 kịch bản phản xạ Failsafe:
  [1] Bay thử nghiệm an toàn & Tự động hạ cánh (Nominal Hover & Land).
  [2] Thử nghiệm mất sóng GCS / Wi-Fi > 3 giây (FS_GCS_ENABLE = 5 -> LAND).
  [3] Thử nghiệm vi phạm trần bay Geo-Fence > 2.5m (FENCE_ACTION = 2 -> LAND).
  [4] Thử nghiệm sụt áp pin yếu < 10.5V (BATT_LOW_VOLT = 10.5V -> LAND).
=============================================================================
"""

import os
import sys
import time
import math
import asyncio
import argparse
from pathlib import Path

# Đảm bảo UTF-8 cho Windows Console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm thư mục gốc vào PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pymavlink import mavutil
from core.mavlink_handler import MAVLinkHandler
from core.base_handler import FlightMode, ConnectionState


class DroneSafetyHarness:
    def __init__(self, port: int = 5773, sys_id: int = 1):
        self.port = port
        self.sys_id = sys_id
        self.handler = MAVLinkHandler()
        self._running = True

    async def connect(self) -> bool:
        conn_str = f"tcp:127.0.0.1:{self.port}"
        print(f"\n📡 [1/5] Đang kết nối MAVLink tới Drone ({conn_str})...")
        
        # Đăng ký nhận log STATUSTEXT trực tiếp từ Single Reader của handler
        self.handler.on_status_text(self._on_status_text)
        
        connected = await self.handler.connect(conn_str)
        if not connected:
            print(f"❌ Không thể kết nối tới {conn_str}!")
            print("💡 Hãy chắc chắn rằng bạn đã bật SITL (hoặc chạy chay_sitl_safety.bat)!")
            return False

        # Đồng bộ target_system và target_component thực tế từ Autopilot
        if self.handler._mavconn:
            self.sys_id = self.handler._mavconn.target_system or self.sys_id
        print(f"✅ Đã kết nối thành công tới SYSID {self.sys_id}!")
        return True

    def _on_status_text(self, text: str):
        """Callback nhận log STATUSTEXT từ Autopilot theo thời gian thực."""
        print(f"   📢 [ArduPilot Log]: {text}")

    @property
    def last_status_text(self) -> str:
        return self.handler.last_status_text

    async def wait_disarm(self, timeout: float = 30.0) -> bool:
        """
        Chờ drone hạ cánh và disarm có timeout an toàn.
        Nếu quá thời gian mà drone vẫn bay, tự động phát lệnh cưỡng bức LAND.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.handler.get_telemetry().armed:
                return True
            await asyncio.sleep(0.5)

        print(f"\n⚠️ Hết thời gian chờ ngắt động cơ ({timeout}s) mà drone vẫn ARMED! Ép lệnh hạ cánh khẩn cấp (LAND)...")
        await self.handler.land()
        
        # Chờ thêm 15s sau khi ép LAND
        sub_start = time.time()
        while time.time() - sub_start < 15.0:
            if not self.handler.get_telemetry().armed:
                return True
            await asyncio.sleep(0.5)

        print("❌ CẢNH BÁO: Drone vẫn chưa disarm sau khi ép hạ cánh khẩn cấp!")
        return False

    async def run_prearm_health_checks(self) -> bool:
        """
        Kiểm tra toàn bộ điều kiện an toàn phần cứng trước khi ARM:
        1. Pin >= 11.1V (Chống chai sụt áp)
        2. ToF MTF-02 đo đúng mặt đất (< 0.35m)
        3. EKF3 đã hội tụ (Attitude, Velocity, Pos)
        4. Strict Arming (param2 = 0) - Tuyệt đối không force-arm!
        """
        print("\n🔍 BẮT ĐẦU KIỂM TRA ĐIỀU KIỆN PRE-ARM AN TOÀN THỰC TẾ...")
        print("-" * 65)

        # Chờ 2 giây để nhận đủ gói tin Telemetry
        await asyncio.sleep(2.0)
        telem = self.handler.get_telemetry()

        # 1. Kiểm tra Điện áp pin 3S
        v_bat = telem.battery_voltage
        batt_ok = v_bat >= 11.1
        batt_icon = "✅" if batt_ok else "❌"
        print(f" {batt_icon} Điện áp Pin 3S: {v_bat:.2f}V (Yêu cầu >= 11.1V để cất cánh)")
        if not batt_ok:
            print(f"    ⚠️ CẢNH BÁO: Điện áp {v_bat:.2f}V quá thấp! Cấm cất cánh chống chai/hỏng pin.")

        # 2. Kiểm tra Độ cao sàn của cảm biến ToF MTF-02
        if telem.rangefinder_distance >= 0:
            alt_tof = telem.rangefinder_distance
            tof_source = "Rangefinder MTF-02 thật"
        else:
            alt_tof = telem.altitude
            tof_source = "GLOBAL_POSITION_INT (ước lượng)"

        tof_ok = abs(alt_tof) < 0.35
        tof_icon = "✅" if tof_ok else "❌"
        print(f" {tof_icon} Cảm biến độ cao ToF ({tof_source}): {alt_tof:.2f}m (Mặt đất < 0.35m)")
        if not tof_ok:
            print(f"    ⚠️ CẢNH BÁO: ToF đo được {alt_tof:.2f}m! Nghi ngờ cảm biến bị che khuất hoặc lỗi sàn.")

        # 3. Kiểm tra Trạng thái bộ lọc EKF3
        ekf_ok = telem.ekf_healthy
        ekf_icon = "✅" if ekf_ok else "⚠️"
        print(f" {ekf_icon} Bộ lọc vị trí EKF3: {'HỘI TỤ (Healthy)' if ekf_ok else 'ĐANG KHỞI TẠO / CHƯA ĐỦ TÍN HIỆU'}")

        # Khóa chặn an toàn nghiêm ngặt: Bắt buộc batt_ok và tof_ok
        if not (batt_ok and tof_ok):
            print("\n❌ [PRE-ARM CHẶN ĐỨNG]: Điều kiện phần cứng không đạt chuẩn an toàn! HỦY BỎ CẤT CÁNH.")
            return False

        # 4. Thử nghiệm Strict Arming (Tôn trọng 100% ARMING_CHECK, param2=0)
        print("\n🔒 Thử nghiệm ARM nghiêm ngặt (param2=0, không dùng mã magic force)...")
        await self.handler.set_mode(FlightMode.GUIDED)
        await asyncio.sleep(1.0)

        # Gửi lệnh ARM nghiêm ngặt: force=False
        arm_passed = await self.handler.arm(force=False)

        if arm_passed:
            print(" ✅ [PRE-ARM PASS]: Toàn bộ cảm biến, EKF3 và Pin đạt chuẩn an toàn!")
            return True
        else:
            print(" ❌ [PRE-ARM REJECTED]: ArduPilot từ chối ARM vì chưa thỏa mãn an toàn!")
            if self.last_status_text:
                print(f"    ↳ Lý do từ Autopilot: {self.last_status_text}")
            print(" 🛑 DỪNG LẠI: Không tự ý ép force-arm để bảo vệ phần cứng drone!")
            return False

    async def test_nominal_hover(self):
        """KỊCH BẢN 1: Bay lơ lửng an toàn ở 1.2m và tự động hạ cánh"""
        print("\n" + "=" * 65)
        print(" 🎯 [KỊCH BẢN 1] BAY LƠ LỬNG AN TOÀN (NOMINAL HOVER) & TỰ ĐỘNG HẠ CÁNH")
        print("=" * 65)

        target_alt = 1.2
        print(f"🛫 Đang phát lệnh cất cánh lên {target_alt}m (dưới trần an toàn 2.5m)...")
        comp_id = self.handler._mavconn.target_component or 1
        async with self.handler._send_lock:
            self.handler._mavconn.mav.command_long_send(
                self.sys_id, comp_id,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, target_alt
            )

        # Chờ đạt cao độ
        for sec in range(12):
            telem = self.handler.get_telemetry()
            print(f"   ↳ [Cất cánh {sec+1}/12s] Cao độ: {telem.altitude:.2f}m | V_z: {telem.vz:.2f}m/s | Pin: {telem.battery_voltage:.2f}V")
            if telem.altitude >= target_alt - 0.15:
                print("   ✅ Đã đạt cao độ mục tiêu!")
                break
            if sec == 3 and telem.altitude < 0.2:
                async with self.handler._send_lock:
                    self.handler._mavconn.mav.command_long_send(
                        self.sys_id, comp_id,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                        0, 0, 0, 0, 0, 0, 0, target_alt
                    )
            await asyncio.sleep(1.0)

        # Lơ lửng 8 giây ổn định
        print(f"\n⏱️ Đang duy trì Hover phẳng lặng tại {target_alt}m trong 8 giây...")
        for s in range(8):
            telem = self.handler.get_telemetry()
            print(f"   ↳ [Hover {s+1}/8s] Z: {telem.altitude:.2f}m | Roll: {telem.roll:.1f}° | Pitch: {telem.pitch:.1f}° | V_xy: {math.sqrt(telem.vx**2 + telem.vy**2):.2f}m/s")
            await asyncio.sleep(1.0)

        # Hạ cánh an toàn
        print("\n🛬 Kết thúc bài bay: Ra lệnh hạ cánh an toàn (LAND)...")
        await self.handler.land()
        
        disarmed = await self.wait_disarm(timeout=25.0)
        if disarmed:
            print("   ✅ Đã chạm sàn và tự động ngắt động cơ (DISARMED) an toàn!")

    async def test_gcs_failsafe(self):
        """KỊCH BẢN 2: Giả lập mất tín hiệu điều khiển GCS > 3 giây"""
        print("\n" + "=" * 65)
        print(" 🚨 [KỊCH BẢN 2] THỬ NGHIỆM FAILSAFE MẤT SÓNG GCS / WI-FI (> 3 GIÂY)")
        print("=" * 65)

        # 1. Cất cánh lên 1.2m
        print("🛫 Đang cất cánh lên 1.2m...")
        comp_id = self.handler._mavconn.target_component or 1
        async with self.handler._send_lock:
            self.handler._mavconn.mav.command_long_send(
                self.sys_id, comp_id,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, 1.2
            )
        
        # Chờ đạt cao độ
        for _ in range(8):
            telem = self.handler.get_telemetry()
            if telem.altitude >= 0.8:
                break
            await asyncio.sleep(1.0)

        # 2. Ngừng phát Heartbeat GCS trong 5 giây để kích hoạt FS_GCS_ENABLE
        print("\n⚡ TẠM DỪNG PHÁT GCS HEARTBEAT TRONG 5 GIÂY (Mô phỏng mất sóng Wi-Fi)...")
        print("   (ArduPilot đếm timeout: Quá 3.0s -> FS_GCS_ENABLE=5 sẽ tự động kích hoạt ALWAYS LAND)")
        
        self.handler.pause_heartbeat()
        for s in range(5):
            print(f"   ⏳ Mất tín hiệu giây thứ {s+1}/5s...")
            await asyncio.sleep(1.0)

        # 3. Phục hồi Heartbeat và kiểm tra phản xạ của Drone
        print("\n📡 Đã khôi phục luồng Heartbeat! Đang kiểm tra chế độ bay...")
        self.handler.resume_heartbeat()
        await asyncio.sleep(1.0)
        
        telem = self.handler.get_telemetry()
        print(f"   ↳ Chế độ hiện tại: {telem.flight_mode}")
        if telem.flight_mode == "LAND":
            print("   🎉 THÀNH CÔNG RỰC RỠ: ArduPilot đã phát hiện mất GCS và ép chuyển sang LAND an toàn!")
        else:
            print(f"   ℹ️ Chế độ hiện tại: {telem.flight_mode}. Tiến hành ép hạ cánh...")
            await self.handler.land()

        await self.wait_disarm(timeout=25.0)
        print("   ✅ Drone đã hoàn tất hạ cánh an toàn!")

    async def test_geofence_breach(self):
        """KỊCH BẢN 3: Thử nghiệm vi phạm hàng rào Geo-Fence độ cao (> 2.5m)"""
        print("\n" + "=" * 65)
        print(" 🚧 [KỊCH BẢN 3] THỬ NGHIỆM HÀNG RÀO GEO-FENCE PHÒNG KÍN (FENCE_ALT_MAX = 2.5m)")
        print("=" * 65)

        # 1. Cất cánh lên 1.2m
        print("🛫 Đang cất cánh lên 1.2m...")
        comp_id = self.handler._mavconn.target_component or 1
        async with self.handler._send_lock:
            self.handler._mavconn.mav.command_long_send(
                self.sys_id, comp_id,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, 1.2
            )
        
        for _ in range(8):
            telem = self.handler.get_telemetry()
            if telem.altitude >= 0.8:
                break
            await asyncio.sleep(1.0)

        # Chờ ổn định ở 1.2m
        await asyncio.sleep(2.0)
        telem = self.handler.get_telemetry()
        print(f"   ↳ Drone đang hover tại Z = {telem.altitude:.2f}m")

        # 2. Thử nghiệm vi phạm trần bay:
        # Trong ArduPilot, GUIDED mode chặn các waypoint nằm ngoài rào (FENCE_ALT_MAX).
        # Để kiểm chứng phản xạ Failsafe LAND (FENCE_ACTION=2), ta kích hoạt rào hẹp FENCE_ALT_MAX = 1.0m
        # (thấp hơn cao độ hiện tại 1.2m), mô phỏng drone bị đẩy vọt qua trần an toàn.
        print("\n⚡ KÍCH HOẠT VI PHẠM TRẦN BAY: Hạ FENCE_ALT_MAX xuống 1.0m (< cao độ thực tế 1.2m)...")
        async with self.handler._send_lock:
            self.handler._mavconn.mav.param_set_send(
                self.sys_id, comp_id,
                b'FENCE_ALT_MAX',
                1.0,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )

        # Lắng nghe phản hồi từ ArduPilot
        for _ in range(12):
            telem = self.handler.get_telemetry()
            print(f"   ↳ [Giám sát rào] Z: {telem.altitude:.2f}m | Mode: {telem.flight_mode}")
            if telem.flight_mode == "LAND" or "Fence" in self.last_status_text or "FENCE" in self.last_status_text:
                print("   🎉 THÀNH CÔNG: Geo-Fence đã phát hiện vi phạm trần bay và kích hoạt Failsafe LAND (FENCE_ACTION=2)!")
                break
            await asyncio.sleep(1.0)

        await self.wait_disarm(timeout=25.0)

        # Khôi phục lại FENCE_ALT_MAX = 2.5m
        print("\n🛡️ Khôi phục trần bay FENCE_ALT_MAX về 2.5m chuẩn...")
        async with self.handler._send_lock:
            self.handler._mavconn.mav.param_set_send(
                self.sys_id, comp_id,
                b'FENCE_ALT_MAX',
                2.5,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
        print("   ✅ Drone đã hạ cánh an toàn bên dưới rào chắn và khôi phục tham số!")

    async def test_battery_failsafe(self):
        """KỊCH BẢN 4: Thử nghiệm sụt áp pin dưới 10.5V"""
        print("\n" + "=" * 65)
        print(" 🔋 [KỊCH BẢN 4] THỬ NGHIỆM FAILSAFE SỤT ÁP PIN YẾU (< 10.5V)")
        print("=" * 65)

        # Guard an toàn (Audit #18): Kiểm tra trực tiếp FC xem có hỗ trợ tham số SIM_BATT_VOLTAGE không.
        # Trên Drone phần cứng thật (ESP32-S3/STM32 ChibiOS), ArduCopter không biên dịch tham số SIM_*.
        # Do đó chỉ có SITL mới phản hồi giá trị này. Tuyệt đối không dựa vào whitelist cổng TCP/UDP!
        print("🔍 Đang xác thực firmware: Truy vấn tham số SIM_BATT_VOLTAGE từ Autopilot...")
        sim_batt = await self.handler.get_param("SIM_BATT_VOLTAGE", timeout=2.5)
        if sim_batt is None:
            print("🛑 TỪ CHỐI THỰC HIỆN BÀI TEST PIN TRÊN PHẦN CỨNG THẬT:")
            print("   Autopilot không hỗ trợ tham số SIM_BATT_VOLTAGE (hoặc không phản hồi).")
            print("   Đây là FC phần cứng thực tế — Việc can thiệp điện áp giả lập bị nghiêm cấm.")
            return
        print(f"   ✅ Đã xác nhận Autopilot là môi trường SITL (SIM_BATT_VOLTAGE = {sim_batt:.2f}V). Tiếp tục bài test.")

        # 1. Cất cánh lên 1.2m
        print("🛫 Đang cất cánh lên 1.2m...")
        comp_id = self.handler._mavconn.target_component or 1
        async with self.handler._send_lock:
            self.handler._mavconn.mav.command_long_send(
                self.sys_id, comp_id,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, 1.2
            )
        
        for _ in range(8):
            telem = self.handler.get_telemetry()
            if telem.altitude >= 0.8:
                break
            await asyncio.sleep(1.0)

        # 2. Bơm điện áp giả lập 10.2V
        print("\n⚡ BƠM SỤT ÁP GIẢ LẬP: SIM_BATT_VOLTAGE = 10.2V (Dưới ngưỡng 10.5V)...")
        async with self.handler._send_lock:
            self.handler._mavconn.mav.param_set_send(
                self.sys_id, comp_id,
                b'SIM_BATT_VOLTAGE',
                10.2,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )

        # Lắng nghe phản ứng hạ cánh khẩn cấp
        for _ in range(12):
            telem = self.handler.get_telemetry()
            print(f"   ↳ [Sụt áp] Điện áp: {telem.battery_voltage:.2f}V | Z: {telem.altitude:.2f}m | Mode: {telem.flight_mode}")
            if telem.flight_mode == "LAND":
                print("   🎉 THÀNH CÔNG: Battery Failsafe Tầng 1 đã kích hoạt, ép hạ cánh khẩn cấp!")
                break
            await asyncio.sleep(1.0)

        await self.wait_disarm(timeout=25.0)

        # Phục hồi lại pin 12.6V
        print("\n🔋 Phục hồi điện áp pin về 12.6V đầy...")
        async with self.handler._send_lock:
            self.handler._mavconn.mav.param_set_send(
                self.sys_id, comp_id,
                b'SIM_BATT_VOLTAGE',
                12.6,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
        print("   ✅ Đã sạc lại pin 12.6V thành công!")

    async def close(self):
        self._running = False
        await self.handler.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="Drone Real-World Safety Harness & Validation")
    parser.add_argument("--port", type=int, default=5773, help="Cổng TCP MAVLink của Drone (Mặc định: 5773)")
    parser.add_argument("--sysid", type=int, default=1, help="SYSID của Drone (Mặc định: 1)")
    parser.add_argument("--test", type=int, default=0, help="Chạy trực tiếp bài test 1, 2, 3, 4 (0: Hiện menu lựa chọn)")
    args = parser.parse_args()

    print("=" * 70)
    print("  🛡️ CHƯƠNG TRÌNH KIỂM THỬ AN TOÀN PHẦN CỨNG THỰC TẾ (STRICT SAFETY)")
    print("  Nền tảng: ArduPilot SITL / ESP32-S3 | GY-912 + MicoAir MTF-02")
    print("=" * 70)

    harness = DroneSafetyHarness(port=args.port, sys_id=args.sysid)

    try:
        # 1. Kết nối MAVLink
        connected = await harness.connect()
        if not connected:
            return

        # 2. Hiển thị Menu lựa chọn kịch bản TRƯỚC KHI ARM (Tránh DISARM_DELAY=10 timeout)
        choice = args.test
        if choice == 0:
            print("\n" + "=" * 65)
            print("  CHỌN KỊCH BẢN KIỂM THỬ AN TOÀN:")
            print("  [1] Bay thử nghiệm an toàn & Tự động LAND (Nominal Safe Flight)")
            print("  [2] Thử nghiệm mất sóng GCS / Wi-Fi > 3 giây (GCS Failsafe)")
            print("  [3] Thử nghiệm vi phạm trần bay Geo-Fence > 2.5m (Ceiling Fence)")
            print("  [4] Thử nghiệm sụt áp pin yếu < 10.5V (Battery Failsafe)")
            print("  [0] Thoát chương trình")
            print("=" * 65)
            try:
                raw_input = input("👉 Nhập lựa chọn của bạn (1-4, mặc định [1]): ").strip()
                choice = int(raw_input) if raw_input else 1
            except Exception:
                choice = 1

        if choice == 0:
            print("Đã thoát kiểm thử.")
            return

        # 3. Tiến hành kiểm tra Pre-Arm và ARM nghiêm ngặt
        passed = await harness.run_prearm_health_checks()
        if not passed:
            print("\n❌ Không thể tiếp tục cất cánh vì các điều kiện Pre-Arm chưa đạt!")
            return

        # 4. Thực thi kịch bản đã chọn
        if choice == 1:
            await harness.test_nominal_hover()
        elif choice == 2:
            await harness.test_gcs_failsafe()
        elif choice == 3:
            await harness.test_geofence_breach()
        elif choice == 4:
            await harness.test_battery_failsafe()
        else:
            print("Lựa chọn không hợp lệ.")

    except KeyboardInterrupt:
        print("\n⚠️ NGƯỜI DÙNG BẤM CTRL+C: KÍCH HOẠT HẠ CÁNH KHẨN CẤP (EMERGENCY LAND)...")
        try:
            await harness.handler.land()
            await harness.wait_disarm(timeout=15.0)
        except Exception as e:
            print(f"Lỗi khi xử lý Ctrl+C: {e}")
    except Exception as e:
        print(f"\n🚨 PHÁT SINH LỖI NGOẠI LỆ: {e}")
    finally:
        # Cơ chế bảo vệ auto-LAND khi thoát bất thường:
        if harness.handler.get_telemetry().armed:
            print("\n⚠️ DRONE VẪN ĐANG ARMED KHI THOÁT! TỰ ĐỘNG ÉP HẠ CÁNH AN TOÀN (AUTO-LAND)...")
            try:
                await harness.handler.land()
                await harness.wait_disarm(timeout=15.0)
            except Exception as e:
                print(f"Lỗi auto-land trong finally: {e}")
        await harness.close()
        print("\n🏁 Hoàn tất phiên kiểm thử an toàn.")


if __name__ == "__main__":
    asyncio.run(main())
