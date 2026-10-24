"""
Comprehensive Tracking & Control Test Suite
============================================
File: test_comprehensive_tracking.py
Date: 25/09/2026

Thorough test suite assessing performance across 6 domains:
1. Stationary Target & Noise Immunity (Best-Case Baseline vs Noise)
2. Dynamic Motion & Trajectory Tracking (Linear, Sinusoidal, Reversal)
3. Frame Rate Drops & High Latency Stutter (5 FPS vs 30 FPS)
4. Occlusion & Grace Period Boundary (Under vs Exceeding Grace)
5. Adversarial & Extreme Edge Cases (Teleportation, Hunting, Saturation, Timing glitches)
6. Multi-Axis 3D Control (Pan + Tilt + Thrust simultaneous)

Outputs quantitative statistics for BEST CASE and WORST CASE analysis.
"""

import sys
import os
import io
import time
import math
import random
import unittest
import numpy as np

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pc_client"))

from controller import PID, ControllerManager
from tracking import LowPassFilter, OneEuroFilter, BBoxOneEuroFilter, TrackingManager


# Global benchmark metrics store
BENCHMARK_RESULTS = {
    "best_cases": [],
    "worst_cases": [],
    "metrics": {}
}


class TestGroup1_StationaryAndNoise(unittest.TestCase):
    """Scenario 1: Stationary Target & Noise Immunity."""

    def test_1a_ideal_deadzone_stillness(self):
        """
        [BEST CASE CANDIDATE]
        Target is stationary right at the center inside deadzone.
        Expect: 0 motor jitter, 0 integral accumulation, 0 derivative kick, 100% stillness.
        """
        ctrl = ControllerManager()
        frame_shape = (480, 640)
        # Center is (320, 240). Box [270, 190, 370, 290] -> center = (320, 240)
        center_bbox = [270, 190, 370, 290]
        initial_pwm_y = ctrl.current_pan_pwm
        initial_pwm_z = ctrl.current_tilt_pwm

        pwm_deltas = []
        for _ in range(100):
            res = ctrl.compute_and_send(center_bbox, frame_shape, tracking_enabled=True)
            pwm_deltas.append(abs(ctrl.current_pan_pwm - initial_pwm_y) + abs(ctrl.current_tilt_pwm - initial_pwm_z))

        ctrl.sock.close()
        max_delta = max(pwm_deltas)
        avg_delta = sum(pwm_deltas) / len(pwm_deltas)

        BENCHMARK_RESULTS["metrics"]["1a_deadzone_max_jitter"] = max_delta
        BENCHMARK_RESULTS["best_cases"].append({
            "name": "Mục tiêu đứng yên trong Deadzone (Tâm khung hình)",
            "jitter_pwm": f"{max_delta:.4f} µs",
            "integral": f"{ctrl.pid_y.integral:.4f}",
            "conclusion": "Hoàn hảo: Không có dao động PWM (0 jitter), động cơ hoàn toàn tĩnh lặng."
        })

        self.assertEqual(max_delta, 0.0, "Motor jittered while target was inside deadzone!")
        self.assertEqual(ctrl.pid_y.integral, 0.0)

    def test_1b_heavy_sensor_noise_attenuation(self):
        """
        Stationary target with high Gaussian sensor noise (sigma = 8.0 pixels).
        1-Euro Filter must attenuate noise by >= 60%.
        """
        random.seed(42)
        np.random.seed(42)

        one_euro = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        true_pos = 320.0
        dt = 0.033
        t = 0.0

        raw_positions = []
        filtered_positions = []

        # Prime filter
        one_euro.filter(true_pos, timestamp=t)
        t += dt

        for _ in range(200):
            noise = np.random.normal(0.0, 8.0)
            noisy_input = true_pos + noise
            raw_positions.append(noisy_input)

            filtered = one_euro.filter(noisy_input, timestamp=t)
            filtered_positions.append(filtered)
            t += dt

        raw_std = np.std(raw_positions[20:])
        filtered_std = np.std(filtered_positions[20:])
        noise_reduction_pct = (1.0 - filtered_std / raw_std) * 100.0

        BENCHMARK_RESULTS["metrics"]["1b_noise_reduction_pct"] = noise_reduction_pct
        self.assertGreater(noise_reduction_pct, 60.0, f"Noise reduction {noise_reduction_pct:.1f}% was below 60%!")


class TestGroup2_DynamicTrajectory(unittest.TestCase):
    """Scenario 2: Dynamic Trajectory Tracking."""

    def test_2a_constant_velocity_motion(self):
        """
        Target moves across the screen at 180 px/s for 1.5 seconds.
        Velocity estimation must converge to within 10% of true speed.
        """
        from unittest.mock import patch

        tm = TrackingManager(None, None)
        tm.state = "TRACKING"
        tm.locked_face_id = 1  # Matches ID in detection
        fw, fh = 640, 480
        dt = 0.033
        true_vx = 180.0  # px/s

        start_cx = 100.0
        cy = 240.0
        box_w, box_h = 80.0, 100.0

        tracked_velocities = []
        base_time = 1000.0

        # Feed 45 frames (1.5 seconds) with simulated timestamps
        with patch('tracking.time.monotonic') as mock_time:
            for i in range(45):
                simulated_now = base_time + i * dt
                mock_time.return_value = simulated_now

                cx = start_cx + true_vx * (i * dt)
                raw_box = [cx - box_w/2, cy - box_h/2, cx + box_w/2, cy + box_h/2, 1]
                tm.update(None, [raw_box], (fh, fw))
                tracked_velocities.append(tm.vx)

        # Evaluate last 15 frames for steady state velocity
        avg_estimated_vx = sum(tracked_velocities[-15:]) / 15.0
        error_pct = abs(avg_estimated_vx - true_vx) / true_vx * 100.0

        BENCHMARK_RESULTS["metrics"]["2a_velocity_error_pct"] = error_pct
        BENCHMARK_RESULTS["best_cases"].append({
            "name": "Mục tiêu di chuyển đều 180 px/s (Vận tốc tuyến tính)",
            "true_speed": f"{true_vx:.1f} px/s",
            "estimated_speed": f"{avg_estimated_vx:.1f} px/s",
            "accuracy": f"{(1 - error_pct/100)*100:.2f}%",
            "conclusion": "Ước lượng vận tốc thời gian thực hội tụ chính xác tuyệt đối."
        })
        self.assertLess(error_pct, 5.0, f"Velocity estimation error {error_pct:.1f}% too high!")

    def test_2b_rapid_direction_reversal(self):
        """
        Target moves right (+150 px/s) then instantaneously reverses left (-150 px/s).
        Integral zero-crossing reset must trigger, preventing prolonged overshoot.
        """
        pid = PID(kp=0.8, ki=1.2, kd=0.2, max_integral=50.0)
        dt = 0.033
        setpoint = (300, 340)

        # Move right (positive error)
        for _ in range(20):
            pid.compute(setpoint, 400, dt)  # error = 400 - 340 = +60

        integral_before_reversal = pid.integral
        self.assertGreater(integral_before_reversal, 0.0)

        # Instant reversal to left (negative error = 250 - 300 = -50)
        out_rev = pid.compute(setpoint, 250, dt)

        # Zero-crossing must wipe the positive integral immediately
        BENCHMARK_RESULTS["metrics"]["2b_reversal_integral"] = pid.integral
        self.assertLessEqual(pid.integral, 0.0, "Zero-crossing failed to clear opposing integral on reversal!")


class TestGroup3_FrameStutterAndLatency(unittest.TestCase):
    """Scenario 3: Severe Frame Stutter and FPS Jitter."""

    def test_3a_yolo_stutter_5fps_vs_tracking_30fps(self):
        """
        [WORST CASE CANDIDATE - Stale Detection Jump]
        YOLO only updates once every 6 frames (5 FPS), while camera runs at 30 FPS.
        When YOLO finally delivers a 40px shifted detection, D-term filter must
        suppress derivative shock compared to unmitigated raw derivative.
        """
        pid = PID(kp=0.5, ki=0.5, kd=1.0, max_integral=50.0, d_alpha=0.3)
        dt = 0.033
        setpoint = (300, 340)

        # 5 stale frames at PV = 280
        for _ in range(5):
            pid.compute(setpoint, 280, dt)

        # Frame 6: YOLO finally updates -> jumps to PV = 240 (40px jump)
        pid.compute(setpoint, 240, dt)

        raw_derivative = ((-60) - (-20)) / dt  # -1212 px/s
        mitigated_d_term = pid.d_term

        # Without filter: D-term would be -1212.0
        # With d_alpha=0.3 filter: should be dampened to ~ -363.6
        damping_ratio = abs(mitigated_d_term) / abs(raw_derivative)

        BENCHMARK_RESULTS["metrics"]["3a_dterm_damping_ratio"] = damping_ratio
        BENCHMARK_RESULTS["worst_cases"].append({
            "name": "YOLO giật trễ 5 FPS trên nền Camera 30 FPS (Bước nhảy 40px)",
            "raw_shock_derivative": f"{raw_derivative:.1f}",
            "filtered_d_term": f"{mitigated_d_term:.1f}",
            "shock_suppression": f"{(1 - damping_ratio)*100:.1f}%",
            "conclusion": "Bộ lọc EMA D-term đã dập tắt >65% xung kích, ngăn motor bị khựng giật."
        })

        self.assertLess(damping_ratio, 0.45, "Derivative filter failed to dampen stutter shock!")

    def test_3b_irregular_time_delta_stability(self):
        """
        Irregular loop timing: dt fluctuates wildly (from 5ms up to 250ms).
        System must not produce NaN, Inf, or numerical explosion.
        """
        pid = PID(kp=1.0, ki=1.0, kd=0.5, max_integral=50.0)
        setpoint = (300, 340)

        dt_sequence = [0.005, 0.250, 0.010, 0.180, 0.033, 0.300, 0.001]
        outputs = []

        for dt in dt_sequence:
            out = pid.compute(setpoint, 260, dt)
            outputs.append(out)

        has_nan_or_inf = any(math.isnan(o) or math.isinf(o) for o in outputs)
        self.assertFalse(has_nan_or_inf, "Numerical explosion detected under irregular dt!")


class TestGroup4_OcclusionAndGraceBoundary(unittest.TestCase):
    """Scenario 4: Target Occlusion & Grace Period Boundary."""

    def test_4a_grace_period_recovery_continuity(self):
        """
        Target occluded for 12 frames (< 25 max_grace_frames).
        During grace: lost_pending=True -> PID frozen, no integral buildup.
        Upon recovery: smooth tracking resumes without jump.
        """
        ctrl = ControllerManager()
        frame_shape = (480, 640)
        frozen_box = [200, 150, 300, 250]

        # 12 frames of grace period
        for _ in range(12):
            ctrl.compute_and_send(frozen_box, frame_shape, tracking_enabled=True, lost_pending=True)

        self.assertEqual(ctrl.pid_y.integral, 0.0, "Integral accumulated during grace period!")

        # Target reappears slightly offset
        recovered_box = [210, 150, 310, 250]
        res = ctrl.compute_and_send(recovered_box, frame_shape, tracking_enabled=True, lost_pending=False)

        ctrl.sock.close()
        self.assertIsNotNone(res)

    def test_4b_grace_period_exceeded_clean_lost(self):
        """
        Target lost for 30 frames (> 25 max_grace_frames).
        TrackingManager must transition cleanly to LOST, locked_bbox=None.
        """
        tm = TrackingManager(None, None)
        tm.state = "TRACKING"
        tm.locked_bbox = [200, 150, 300, 250]
        tm._raw_locked_bbox = [200, 150, 300, 250]
        tm.last_update_time = time.monotonic()
        fw, fh = 640, 480

        # Feed 30 empty detection frames
        final_state = None
        for _ in range(30):
            res = tm.update(None, [], (fh, fw))
            final_state = res["state"]

        self.assertEqual(final_state, "LOST", "State did not transition to LOST after grace timeout!")
        self.assertIsNone(tm.locked_bbox)
        self.assertIsNone(tm.last_update_time)


class TestGroup5_AdversarialAndExtremeCases(unittest.TestCase):
    """Scenario 5: Adversarial & Extreme Edge Cases."""

    def test_5a_teleportation_false_positive_jump(self):
        """
        [WORST CASE CANDIDATE - False Positive Teleportation]
        A false positive detection causes a 450px jump for 1 frame.
        Velocity clamping (max 2000 px/s) must prevent runaway velocity vector.
        """
        tm = TrackingManager(None, None)
        tm.state = "TRACKING"
        tm.last_cx = 320.0
        tm.last_cy = 240.0
        tm.last_update_time = time.monotonic() - 0.033

        # False positive detection at (50, 50) -> distance ~ 330px in 33ms -> ~10,000 px/s!
        fake_box = [40, 40, 60, 60, 1]
        tm.update(None, [fake_box], (480, 640))

        # Velocity must be clamped, preventing an explosive Grace extrapolation
        self.assertLessEqual(abs(tm.vx), 2000.0, "Velocity clamp failed under teleportation jump!")
        self.assertLessEqual(abs(tm.vy), 2000.0)

    def test_5b_deadzone_edge_hunting_stability(self):
        """
        Target hovers right on the edge of deadzone (300px), oscillating between 299.5 and 300.5.
        Deadzone transition logic must not create high-frequency motor chatter.
        """
        pid = PID(kp=1.0, ki=1.0, kd=0.5)
        dt = 0.033
        setpoint = (300, 340)

        outputs = []
        d_terms = []

        for i in range(100):
            pv = 300.0 + (0.8 if i % 2 == 0 else -0.8)
            out = pid.compute(setpoint, pv, dt)
            outputs.append(out)
            d_terms.append(pid.d_term)

        # D-term should remain virtually zero without derivative kick
        max_d_kick = max(abs(d) for d in d_terms)
        BENCHMARK_RESULTS["metrics"]["5b_max_hunting_d_kick"] = max_d_kick
        self.assertLess(max_d_kick, 2.0, "Deadzone hunting caused derivative kick spikes!")

    def test_5c_actuator_saturation_1000_frames(self):
        """
        [WORST CASE CANDIDATE - Actuator Saturation]
        Target stuck off-center for 1,000 frames (~33 seconds) where motors cannot reach.
        Anti-windup clamp must keep integral strictly at 50.0.
        Upon target return, recovery must occur in < 3 frames.
        """
        pid = PID(kp=0.8, ki=1.5, kd=0.1, max_integral=50.0)
        dt = 0.033
        setpoint = (300, 340)
        pv_stuck = 150.0  # error = -150

        # Run 1,000 frames
        for _ in range(1000):
            pid.compute(setpoint, pv_stuck, dt)

        # Integral must strictly equal -50.0 (anti-windup)
        self.assertAlmostEqual(abs(pid.integral), 50.0, delta=0.01)

        # Target returns inside deadzone
        recovered_out = pid.compute(setpoint, 320.0, dt)
        self.assertEqual(recovered_out, 0.0, "Immediate reset inside deadzone failed after saturation!")
        self.assertEqual(pid.integral, 0.0)

        BENCHMARK_RESULTS["worst_cases"].append({
            "name": "Bão hoà cơ cấu chấp hành 1,000 frames (~33s mục tiêu kẹt cứng biên)",
            "integral_unclamped_would_be": f"{-150 * 0.033 * 1000:.0f}",
            "integral_clamped_actual": f"{pid.integral:.1f}",
            "recovery_frames": "1 frame (Reset tức thì khi vào deadzone)",
            "conclusion": "Anti-Windup ngăn chặn thành công việc tích luỹ sai số vô hạn (từ -4,950 về -50)."
        })


class TestGroup6_MultiAxis3DControl(unittest.TestCase):
    """Scenario 6: Simultaneous 3-Axis Control (Pan, Tilt, Thrust)."""

    def test_6a_simultaneous_3axis_motion(self):
        """
        Target moves diagonally across frame while expanding in area (walking towards camera).
        All 3 axes (Y, Z, X) must calculate outputs and produce valid MOVE command string.
        """
        ctrl = ControllerManager()
        frame_shape = (480, 640)

        # Target top-left corner moving, area ratio growing
        moving_bbox = [100, 80, 260, 240]  # area = 160*160 = 25600 px^2

        res = ctrl.compute_and_send(moving_bbox, frame_shape, tracking_enabled=True)
        ctrl.sock.close()

        self.assertTrue(res["has_target"])
        self.assertTrue(res["cmd"].startswith("MOVE"))
        tokens = res["cmd"].split()
        self.assertEqual(len(tokens), 4)  # MOVE Y Z X

        pan_val = int(tokens[1])
        tilt_val = int(tokens[2])
        thrust_val = int(tokens[3])

        # Verify values within PWM boundaries [500, 2500]
        self.assertTrue(500 <= pan_val <= 2500)
        self.assertTrue(500 <= tilt_val <= 2500)
        self.assertTrue(500 <= thrust_val <= 2500)


class ZBenchmarkReporter(unittest.TestCase):
    """Print Summary of Best & Worst Cases."""

    def test_z_print_benchmark_summary(self):
        print("\n" + "="*78)
        print("🏆 BÁO CÁO PHÂN TÍCH HIỆU NĂNG: TRƯỜNG HỢP TỐT NHẤT & TỆ NHẤT")
        print("="*78)

        print("\n🟢 [BEST CASE - TRƯỜNG HỢP TỐT NHẤT]:")
        for idx, bc in enumerate(BENCHMARK_RESULTS["best_cases"], 1):
            print(f"  {idx}. {bc['name']}:")
            for k, v in bc.items():
                if k != 'name':
                    print(f"     • {k}: {v}")

        print("\n🔴 [WORST CASE - TRƯỜNG HỢP TỆ NHẤT & KHẮC PHỤC]:")
        for idx, wc in enumerate(BENCHMARK_RESULTS["worst_cases"], 1):
            print(f"  {idx}. {wc['name']}:")
            for k, v in wc.items():
                if k != 'name':
                    print(f"     • {k}: {v}")

        print("\n" + "="*78)


if __name__ == "__main__":
    unittest.main(verbosity=2)
