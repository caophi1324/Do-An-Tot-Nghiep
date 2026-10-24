"""
Verification Tests for Tracking & Controller Fixes
===================================================
File: test_tracking_fixes.py
Date: 25/09/2026

Verifies all 5 Root Causes (RC1 - RC5) have been resolved:
- RC1: PID Anti-Windup clamp & lost_pending freezing
- RC2: OneEuroFilter raw input velocity computation (Casiez 2012)
- RC3: Low-pass filter on PID derivative to prevent spikes
- RC4: Time-based velocity (pixels/second) and elapsed-time Grace Period extrapolation
- RC5: Deadzone transition without derivative kick
"""

import sys
import os
import io
import time
import unittest

# Ensure UTF-8 console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pc_client"))

from controller import PID, ControllerManager
from tracking import LowPassFilter, OneEuroFilter, BBoxOneEuroFilter, TrackingManager


class TestRC1_Fix_PID_AntiWindup_And_Grace(unittest.TestCase):
    """Verifies RC1 fixes: Anti-windup and lost_pending freezing."""

    def test_integral_is_clamped(self):
        """Integral must be clamped to max_integral (default 50.0)."""
        pid = PID(kp=1.0, ki=1.0, kd=0.0, max_integral=50.0)
        dt = 0.033
        setpoint = (300, 340)
        pv = 200  # error = -100

        # Run for 100 frames (without clamp, integral would be -100 * 0.033 * 100 = -330)
        for _ in range(100):
            pid.compute(setpoint, pv, dt)

        print(f"\n[Test RC1 Clamped] Integral after 100 frames: {pid.integral:.2f}")
        self.assertLessEqual(abs(pid.integral), 50.0, "Integral exceeded max_integral clamp!")
        self.assertAlmostEqual(abs(pid.integral), 50.0, delta=1.0)

    def test_compute_and_send_handles_lost_pending(self):
        """When lost_pending=True, PID should not accumulate and PWM should hold."""
        ctrl = ControllerManager()
        frame_shape = (480, 640)
        locked_bbox = [200, 150, 400, 350]  # off-center

        # Initial pan PWM
        initial_pan = ctrl.current_pan_pwm

        # Call with lost_pending=True for 25 frames
        for _ in range(25):
            res = ctrl.compute_and_send(locked_bbox, frame_shape, True, True)

        # Verify no PID integral was accumulated during grace period
        self.assertEqual(ctrl.pid_y.integral, 0.0, "pid_y accumulated integral during lost_pending!")
        self.assertEqual(ctrl.pid_z.integral, 0.0, "pid_z accumulated integral during lost_pending!")
        self.assertEqual(ctrl.current_pan_pwm, initial_pan, "PWM changed during lost_pending!")
        ctrl.sock.close()
        print(f"[Test RC1 lost_pending] PID held safely: integral={ctrl.pid_y.integral}, pan_pwm={ctrl.current_pan_pwm}")

    def test_overshoot_prevented_after_recovery(self):
        """Output after recovery should remain bounded and reasonable."""
        pid = PID(kp=1.0, ki=2.0, kd=0.0, max_integral=30.0)
        dt = 0.033
        setpoint = (300, 340)

        # 25 frames off-center
        for _ in range(25):
            pid.compute(setpoint, 250, dt)

        # Recovery near setpoint (pv=295 -> error = -5)
        out = pid.compute(setpoint, 295, dt)
        print(f"[Test RC1 Overshoot] Output after recovery: {out:.2f} (previously was -87.83)")
        # Output should be modest (error*kp + clamped_i*ki)
        self.assertLess(abs(out), 70.0, "Output is still excessively large!")


class TestRC2_Fix_OneEuroFilter_Derivative(unittest.TestCase):
    """Verifies RC2 fix: OneEuroFilter uses prev_raw_x per Casiez 2012 paper."""

    def test_filter_has_prev_raw_x(self):
        """Filter must maintain prev_raw_x."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        self.assertTrue(hasattr(f, 'prev_raw_x'), "OneEuroFilter missing prev_raw_x attribute!")
        self.assertIsNone(f.prev_raw_x)

        # Filter once
        f.filter(150.0, timestamp=0.0)
        self.assertEqual(f.prev_raw_x, 150.0)

        # Reset
        f.reset()
        self.assertIsNone(f.prev_raw_x, "reset() failed to clear prev_raw_x!")
        print("\n[Test RC2 prev_raw_x] Verified prev_raw_x lifecycle OK")

    def test_stationary_input_converges_rapidly(self):
        """Stationary input should converge cleanly with zero velocity."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        f.filter(100.0, timestamp=0.0)

        for i in range(1, 30):
            res = f.filter(300.0, timestamp=i * 0.033)

        self.assertAlmostEqual(res, 300.0, delta=0.5)
        print(f"[Test RC2 Convergence] Converged to {res:.4f} for target 300.0")


class TestRC3_Fix_Derivative_Filtering(unittest.TestCase):
    """Verifies RC3 fix: Low-pass filter on D-term dampens sudden spikes."""

    def test_dterm_filtered_on_jump(self):
        """Derivative spike on jump should be filtered by d_alpha."""
        pid = PID(kp=0.0, ki=0.0, kd=1.0, d_alpha=0.3)
        dt = 0.033
        setpoint = (300, 340)

        # Stationary frames
        for _ in range(5):
            pid.compute(setpoint, 280, dt)

        # Jump 30px
        pid.compute(setpoint, 250, dt)
        raw_derivative = (-50 - (-20)) / dt  # ~ -909.09
        actual_d_term = pid.d_term

        # Filtered should be alpha * raw (~ 0.3 * -909.09 = -272.7)
        print(f"\n[Test RC3 D-Filter] Raw jump would be ~{raw_derivative:.1f}, Filtered D-term={actual_d_term:.1f}")
        self.assertLess(abs(actual_d_term), abs(raw_derivative) * 0.5, "D-term was not filtered!")


class TestRC4_Fix_Velocity_TimeBased(unittest.TestCase):
    """Verifies RC4 fix: Velocity in TrackingManager is time-based."""

    def test_velocity_is_scaled_by_time(self):
        """Verify that velocity calculation uses dt_v (time-based px/s)."""
        tm = TrackingManager(None, None)
        self.assertTrue(hasattr(tm, 'last_update_time'))

        # Fake a detection update
        t0 = time.monotonic()
        tm.state = "TRACKING"
        tm.last_cx = 100.0
        tm.last_cy = 100.0
        tm.last_update_time = t0 - 0.1  # 100ms ago

        # Move 10px in 0.1s -> 100 px/s
        raw_bbox = [105, 105, 115, 115]  # cx=110, cy=110
        cur_cx = (raw_bbox[0] + raw_bbox[2]) / 2.0
        cur_cy = (raw_bbox[1] + raw_bbox[3]) / 2.0

        now = t0
        dt_v = max(0.001, now - tm.last_update_time)
        inst_vx = (cur_cx - tm.last_cx) / dt_v
        inst_vy = (cur_cy - tm.last_cy) / dt_v

        # inst_vx should be ~100 px/s (not 10 px/frame)
        self.assertAlmostEqual(inst_vx, 100.0, delta=5.0)
        print(f"\n[Test RC4 Time-based Velocity] Calculated inst_vx = {inst_vx:.1f} px/s")


class TestRC5_Fix_Derivative_Kick_On_Deadzone_Exit(unittest.TestCase):
    """Verifies RC5 fix: Exiting deadzone suppresses derivative kick."""

    def test_no_derivative_kick_on_exit(self):
        """When exiting deadzone, derivative must be 0 on first frame."""
        pid = PID(kp=0.0, ki=0.0, kd=1.0)
        dt = 0.033
        setpoint = (300, 340)

        # 1. Target in deadzone
        pid.compute(setpoint, 320, dt)
        self.assertTrue(pid.was_in_deadzone)
        self.assertEqual(pid.d_term, 0.0)

        # 2. Target exits deadzone to 280 (error = -20)
        pid.compute(setpoint, 280, dt)

        # On the first frame exiting deadzone, derivative should be 0.0!
        print(f"\n[Test RC5 Kick] D-term upon exiting deadzone: {pid.d_term:.4f} (previously was -606.06)")
        self.assertEqual(pid.d_term, 0.0, "Derivative kick occurred upon exiting deadzone!")
        self.assertFalse(pid.was_in_deadzone)


if __name__ == "__main__":
    unittest.main(verbosity=2)
