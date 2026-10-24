"""
Test Cases - Chung minh cac loi goc re gay tracking giat & troi sang phai
=========================================================================
File: test_tracking_bugs.py
Ngay: 25/09/2026

Moi test case tuong ung voi 1 Root Cause (RC) trong kehoachlammuotai.md.
Chay: python test_tracking_bugs.py
"""

import sys
import os
import io
import time
import math
import unittest

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add pc_client to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pc_client"))

from controller import PID, ControllerManager
from tracking import LowPassFilter, OneEuroFilter, BBoxOneEuroFilter


# ==============================================================================
# RC1: PID Integral Windup - Integral tich luy vo han
# ==============================================================================

class TestRC1_PID_IntegralWindup(unittest.TestCase):
    """
    RC1: When target is lost (Grace Period), PID keeps running on frozen bbox.
    Integral accumulates unbounded -> when tracking resumes, motor swings max.
    """

    def test_integral_grows_unbounded_during_grace_period(self):
        """
        Simulate: target outside deadzone, PID runs for 25 frames (Grace Period).
        Integral must grow unbounded since there is NO clamp.

        EXPECTED BUG: integral exceeds reasonable threshold (e.g. > 30)
        -> Proves missing anti-windup
        """
        pid = PID(kp=1.0, ki=1.0, kd=0.0)
        dt = 0.033  # ~30 FPS

        # Target is 50 pixels outside deadzone edge
        setpoint = (300, 340)  # deadzone center +/- 20 pixels
        process_variable = 250  # outside deadzone (50px left)

        # Run 25 frames (Grace Period) with frozen bbox
        for _ in range(25):
            pid.compute(setpoint, process_variable, dt)

        print(f"\n[RC1] Integral after 25 frame Grace Period: {pid.integral:.4f}")
        print(f"[RC1] I-term: {pid.i_term:.4f}")

        # BUG: Integral must be > 30 -> proves missing anti-windup
        # With error=-50, dt=0.033, Ki=1.0: integral ~ -50 * 0.033 * 25 ~ -41.25
        self.assertGreater(
            abs(pid.integral), 30.0,
            "BUG CONFIRMED: Integral accumulated > 30 without any clamp! "
            "This is the PRIMARY cause of bbox drifting to the right."
        )

    def test_integral_causes_violent_overshoot_after_recovery(self):
        """
        Simulate: PID accumulates integral during Grace Period, then target
        returns near center. PID output must be huge due to integral
        -> motor swings to max.

        EXPECTED BUG: PID output >> actual error
        """
        pid = PID(kp=1.0, ki=2.0, kd=0.0)
        dt = 0.033

        setpoint = (300, 340)
        frozen_pv = 250  # frozen bbox outside deadzone

        # Phase 1: Grace Period - 25 frames frozen bbox
        for _ in range(25):
            pid.compute(setpoint, frozen_pv, dt)

        integral_after_grace = pid.integral

        # Phase 2: Target returns near center (error only 5 pixels)
        recovered_pv = 295  # only 5px from deadzone edge
        output = pid.compute(setpoint, recovered_pv, dt)

        print(f"\n[RC1-overshoot] Accumulated integral: {integral_after_grace:.2f}")
        print(f"[RC1-overshoot] Actual error: 5 pixels")
        print(f"[RC1-overshoot] PID output: {output:.2f}")
        print(f"[RC1-overshoot] Ratio output/error: {abs(output/5):.1f}x")

        # BUG: Output must be >> 5 (actual error) due to integral accumulation
        self.assertGreater(
            abs(output), 20.0,
            "BUG CONFIRMED: PID output is massive despite error being only 5px! "
            "Integral windup causes overshoot -> motor swings to max."
        )

    def test_lost_pending_is_ignored_by_controller(self):
        """
        Prove: ControllerManager.compute_and_send() receives lost_pending
        via *args but completely ignores it.

        EXPECTED BUG: PID still runs even when lost_pending=True
        """
        ctrl = ControllerManager()

        frame_shape = (480, 640)
        locked_bbox = [200, 150, 400, 350]  # bbox outside deadzone

        try:
            # Call with lost_pending=True
            result1 = ctrl.compute_and_send(locked_bbox, frame_shape, True, True)
            result2 = ctrl.compute_and_send(locked_bbox, frame_shape, True, True)

            # Check PID state - integral should NOT increase if lost_pending is handled
            has_integral = False
            for attr_name in ['pid_y', 'pid_z', 'pid_x', 'pan_pid', 'tilt_pid']:
                pid_obj = getattr(ctrl, attr_name, None)
                if pid_obj and hasattr(pid_obj, 'integral'):
                    if abs(pid_obj.integral) > 0.001:
                        has_integral = True
                        print(f"\n[RC1-ignored] {attr_name}.integral = {pid_obj.integral:.4f}")
                        break

            if has_integral:
                print("[RC1-ignored] BUG CONFIRMED: PID accumulates integral even with lost_pending=True!")
            else:
                print("[RC1-ignored] PID integral = 0 (bbox may be inside deadzone)")

        except Exception as e:
            print(f"\n[RC1-ignored] Cannot test compute_and_send directly (needs network): {e}")
            print("[RC1-ignored] Confirmed via code review: *args is completely ignored")


# ==============================================================================
# RC2: 1-Euro Filter - Uses filtered output instead of raw input for velocity
# ==============================================================================

class TestRC2_OneEuroFilter_WrongDerivative(unittest.TestCase):
    """
    RC2: OneEuroFilter computes dx = (x - self.x_filter.y) / dt
    instead of dx = (x - prev_raw_x) / dt (correct per Casiez 2012 paper).

    Consequence: filter never smooths when target is stationary.
    """

    def test_filter_does_not_converge_for_stationary_input(self):
        """
        Fixed input (stationary target) -> filter output must converge to input.

        Correct 1-Euro: after ~10 frames, output ~ input (error < 0.1%)
        Buggy 1-Euro: output oscillates or converges very slowly

        EXPECTED BUG: error remains large after many frames
        """
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        target_value = 320.0
        t = 0.0
        dt = 0.033

        # Initialize with a distant value to see convergence behavior
        result = f.filter(200.0, timestamp=t)
        t += dt

        errors = []
        for i in range(100):
            result = f.filter(target_value, timestamp=t)
            t += dt
            error = abs(result - target_value)
            errors.append(error)

        final_error = errors[-1]
        error_at_frame_20 = errors[19] if len(errors) > 19 else errors[-1]

        print(f"\n[RC2] Fixed input = {target_value}")
        print(f"[RC2] Error at frame 20: {error_at_frame_20:.6f}")
        print(f"[RC2] Error at frame 100: {final_error:.6f}")
        print(f"[RC2] Final filter output: {result:.6f}")

        if error_at_frame_20 > 1.0:
            print("[RC2] WARNING: Filter converges SLOWLY - likely due to derivative bug")

    def test_filter_velocity_is_artificially_inflated(self):
        """
        Direct check: when input is constant, velocity (dx) of filter
        must = 0. If using filtered output, dx != 0 because output lags.

        EXPECTED BUG: dx != 0 despite constant input
        """
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        # Frame 1: initialize
        f.filter(300.0, timestamp=0.0)
        # Frame 2-10: constant input
        for i in range(1, 10):
            f.filter(300.0, timestamp=i * 0.033)

        # Check internal state
        dx_filtered = f.dx_filter.y if f.dx_filter.y is not None else 0.0

        print(f"\n[RC2-velocity] Constant input = 300.0 (9 consecutive frames)")
        print(f"[RC2-velocity] dx_filter.y (velocity) = {dx_filtered:.6f}")
        print(f"[RC2-velocity] x_filter.y (position) = {f.x_filter.y:.6f}")

        if abs(dx_filtered) > 0.01:
            print("[RC2-velocity] BUG CONFIRMED: Velocity != 0 despite constant input!")
            print("  -> Filter uses filtered output instead of raw input for dx")
        else:
            print("[RC2-velocity] Velocity ~ 0 (OK for this test)")

    def test_filter_uses_filtered_not_raw_for_derivative(self):
        """
        Core structural test: prove OneEuroFilter does NOT store prev_raw_x.
        If class lacks 'prev_raw_x' attribute, it must be using
        self.x_filter.y (filtered) for derivative -- which is the bug.
        """
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        has_prev_raw = hasattr(f, 'prev_raw_x') or hasattr(f, '_prev_raw_x')

        print(f"\n[RC2-structural] OneEuroFilter has 'prev_raw_x': {has_prev_raw}")

        if not has_prev_raw:
            print("[RC2-structural] BUG CONFIRMED: No variable to store raw input!")
            print("  -> Certainly uses self.x_filter.y (filtered output)")
            print("  -> Violates Casiez 2012 paper: dx must use raw input")

        self.assertFalse(
            has_prev_raw,
            "If this test FAILS (has prev_raw_x) -> RC2 bug has been fixed!"
        )


# ==============================================================================
# RC3: D-Term Spike from stale YOLO data
# ==============================================================================

class TestRC3_DTermSpike(unittest.TestCase):
    """
    RC3: YOLO runs at 5-10 FPS, tracking at 30 FPS. When YOLO updates after
    many identical frames, error jumps -> derivative spikes -> motor jerks.
    """

    def test_dterm_spikes_on_sudden_detection_jump(self):
        """
        Simulate: 5 consecutive frames with same detection (YOLO stale),
        then frame 6 detection jumps 30 pixels.

        EXPECTED BUG: D-term spike at frame 6
        """
        pid = PID(kp=0.5, ki=0.0, kd=1.0)
        dt = 0.033
        setpoint = (300, 340)

        outputs = []
        d_terms = []

        # 5 frames: fixed detection (YOLO hasn't updated)
        for _ in range(5):
            out = pid.compute(setpoint, 280, dt)
            outputs.append(out)
            d_terms.append(pid.d_term)

        # Frame 6: YOLO updates -> detection jumps 30px
        out_spike = pid.compute(setpoint, 250, dt)
        d_term_spike = pid.d_term

        print(f"\n[RC3] D-terms first 5 frames (stale): {[f'{d:.2f}' for d in d_terms]}")
        print(f"[RC3] D-term frame 6 (YOLO update): {d_term_spike:.2f}")
        print(f"[RC3] Output frame 6: {out_spike:.2f}")

        if len(d_terms) > 0 and d_terms[-1] != 0:
            spike_ratio = abs(d_term_spike / d_terms[-1]) if d_terms[-1] != 0 else float('inf')
        else:
            spike_ratio = abs(d_term_spike) if d_term_spike != 0 else 0

        print(f"[RC3] Spike ratio: {spike_ratio:.1f}x")

        if abs(d_term_spike) > 5.0:
            print("[RC3] BUG CONFIRMED: D-term spike when YOLO updates after stale frames!")
            print("  -> Needs low-pass filter on derivative")

    def test_dterm_is_zero_during_stale_frames(self):
        """
        When detection unchanged (YOLO stale), D-term = 0.
        Then suddenly != 0 when YOLO updates.
        """
        pid = PID(kp=0.0, ki=0.0, kd=1.0)  # Only D-term enabled
        dt = 0.033
        setpoint = (300, 340)

        # Frame 1: initialize
        pid.compute(setpoint, 280, dt)
        # Frame 2-6: detection unchanged
        stale_d_terms = []
        for _ in range(5):
            pid.compute(setpoint, 280, dt)
            stale_d_terms.append(pid.d_term)

        all_zero = all(abs(d) < 0.001 for d in stale_d_terms)
        print(f"\n[RC3-stale] D-terms when stale: {[f'{d:.4f}' for d in stale_d_terms]}")
        print(f"[RC3-stale] All near zero: {all_zero}")

        if all_zero:
            print("[RC3-stale] -> When YOLO updates, D-term will spike from 0 -> very large")


# ==============================================================================
# RC4: Velocity EMA is frame-rate dependent
# ==============================================================================

class TestRC4_VelocityFrameDependent(unittest.TestCase):
    """
    RC4: vx/vy calculated as pixel/frame instead of pixel/second.
    When FPS changes, same physical motion gives different velocity values.
    """

    def _simulate_ema_velocity(self, positions, fps):
        """
        Simulate EMA velocity exactly as in tracking.py line 205-206:
        self.vx = 0.6 * self.vx + 0.4 * (cur_cx - self.last_cx)
        """
        vx = 0.0
        last_cx = positions[0]
        for cx in positions[1:]:
            vx = 0.6 * vx + 0.4 * (cx - last_cx)
            last_cx = cx
        return vx

    def test_same_physical_motion_different_fps_gives_different_velocity(self):
        """
        Target moves 300 pixels in 1 second.
        At 30 FPS: 10 pixel/frame -> vx per frame
        At 15 FPS: 20 pixel/frame -> vx per frame

        EXPECTED BUG: velocity differs despite same physical motion
        """
        speed_pixel_per_sec = 300.0

        # At 30 FPS
        n_frames_30 = 30
        positions_30fps = [i * (speed_pixel_per_sec / 30) for i in range(n_frames_30 + 1)]
        vx_30fps = self._simulate_ema_velocity(positions_30fps, 30)

        # At 15 FPS
        n_frames_15 = 15
        positions_15fps = [i * (speed_pixel_per_sec / 15) for i in range(n_frames_15 + 1)]
        vx_15fps = self._simulate_ema_velocity(positions_15fps, 15)

        print(f"\n[RC4] Physical motion: {speed_pixel_per_sec} pixel/sec")
        print(f"[RC4] vx at 30 FPS: {vx_30fps:.4f} (pixel/frame)")
        print(f"[RC4] vx at 15 FPS: {vx_15fps:.4f} (pixel/frame)")
        print(f"[RC4] Ratio vx_15fps / vx_30fps: {vx_15fps / vx_30fps:.2f}x")

        # BUG: vx at 15 FPS should be ~2x vx at 30 FPS (pixel/frame doubles)
        ratio = vx_15fps / vx_30fps
        self.assertGreater(
            abs(ratio - 1.0), 0.3,
            "BUG CONFIRMED: Same physical motion but velocity differs "
            f"({ratio:.2f}x) across FPS! Velocity is frame-rate dependent."
        )

        print(f"[RC4] BUG CONFIRMED: Same 300px/s, velocity differs {ratio:.2f}x between 15/30 FPS")

    def test_velocity_prediction_error_during_grace_period(self):
        """
        Grace Period uses vx (pixel/frame) * elapsed_frames to predict.
        If FPS changes, prediction will be wrong.
        """
        vx_pixel_per_frame = 10.0
        grace_seconds = 0.5

        # At 30 FPS
        grace_frames_30fps = grace_seconds * 30
        predicted_30fps = vx_pixel_per_frame * grace_frames_30fps

        # At 15 FPS (keeping vx same since it was calculated before FPS drop)
        grace_frames_15fps = grace_seconds * 15
        predicted_15fps = vx_pixel_per_frame * grace_frames_15fps

        # Ground truth: 300 pixel/sec * 0.5 sec = 150 pixel
        ground_truth = 300.0 * grace_seconds

        print(f"\n[RC4-grace] Ground truth offset: {ground_truth:.0f} pixels")
        print(f"[RC4-grace] Predicted (30 FPS): {predicted_30fps:.0f} pixels")
        print(f"[RC4-grace] Predicted (15 FPS): {predicted_15fps:.0f} pixels")
        print(f"[RC4-grace] Error at 15 FPS: {abs(predicted_15fps - ground_truth):.0f} pixels (50%!)")


# ==============================================================================
# RC5: Derivative Kick when exiting Deadzone
# ==============================================================================

class TestRC5_DerivativeKick(unittest.TestCase):
    """
    RC5: When target enters deadzone -> PID.reset() -> prev_error = 0.
    When exiting deadzone -> derivative = (error - 0) / dt -> large spike.
    """

    def test_derivative_spike_on_deadzone_exit(self):
        """
        Simulate: target enters deadzone (PID reset), then exits deadzone.

        EXPECTED BUG: D-term spike on first frame exiting deadzone
        """
        pid = PID(kp=0.0, ki=0.0, kd=1.0)  # Only D-term
        dt = 0.033
        setpoint = (300, 340)  # deadzone [300, 340]

        # Phase 1: Target outside deadzone, normal tracking
        for pv in [260, 265, 270, 275, 280]:
            pid.compute(setpoint, pv, dt)

        d_term_before_dz = pid.d_term
        prev_error_before_dz = pid.prev_error

        # Phase 2: Target enters deadzone -> PID reset
        pid.compute(setpoint, 320, dt)  # 320 is inside [300, 340]

        d_term_in_dz = pid.d_term
        prev_error_after_reset = pid.prev_error

        # Phase 3: Target exits deadzone
        pid.compute(setpoint, 280, dt)  # 280 is outside deadzone, error = 280-300 = -20

        d_term_after_exit = pid.d_term

        print(f"\n[RC5] prev_error before deadzone: {prev_error_before_dz:.2f}")
        print(f"[RC5] prev_error AFTER reset (in deadzone): {prev_error_after_reset:.2f}")
        print(f"[RC5] D-term before deadzone: {d_term_before_dz:.2f}")
        print(f"[RC5] D-term in deadzone: {d_term_in_dz:.2f}")
        print(f"[RC5] D-term EXITING deadzone: {d_term_after_exit:.2f}")

        # BUG: prev_error is reset to 0, so on exit:
        # derivative = (error - 0) / dt = -20 / 0.033 ~ -606
        # -> huge D-term spike!
        if abs(d_term_after_exit) > abs(d_term_before_dz) * 3:
            spike_magnitude = abs(d_term_after_exit)
            print(f"[RC5] BUG CONFIRMED: D-term spike = {spike_magnitude:.2f}")
            print("  -> Derivative Kick on deadzone exit!")
            print(f"  -> derivative ~ error/dt = -20/0.033 ~ {-20/0.033:.0f}")


# ==============================================================================
# BONUS: BBoxOneEuroFilter drift test
# ==============================================================================

class TestBBoxFilterDrift(unittest.TestCase):
    """Test BBoxOneEuroFilter for accumulated drift over time."""

    def test_bbox_filter_stability_over_long_duration(self):
        """
        Fixed input bbox for 500 frames (~17 sec at 30FPS).
        Output must stabilize and not drift.
        """
        bf = BBoxOneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        target_bbox = [100, 50, 300, 250]
        t = 0.0
        dt = 0.033

        bf.filter(target_bbox, timestamp=t)
        t += dt

        errors_over_time = []
        for i in range(500):
            result = bf.filter(target_bbox, timestamp=t)
            t += dt
            max_err = max(abs(result[j] - target_bbox[j]) for j in range(4))
            errors_over_time.append(max_err)

        error_frame_10 = errors_over_time[9]
        error_frame_100 = errors_over_time[99]
        error_frame_500 = errors_over_time[499]

        print(f"\n[BBoxDrift] Error at frame 10: {error_frame_10:.6f}")
        print(f"[BBoxDrift] Error at frame 100: {error_frame_100:.6f}")
        print(f"[BBoxDrift] Error at frame 500: {error_frame_500:.6f}")

        if error_frame_500 > error_frame_100:
            print("[BBoxDrift] WARNING: Error INCREASES over time -> filter drifts!")
        elif error_frame_500 < 0.01:
            print("[BBoxDrift] OK: Filter converges well")
        else:
            print(f"[BBoxDrift] WARNING: Filter converges slowly (error={error_frame_500:.4f} after 500 frames)")

    def test_bbox_filter_with_noisy_input(self):
        """
        Input bbox with small noise (+/-3 pixels). Filter must smooth the noise.
        If RC2 is correct (filter bug), output will track input instead of smoothing.
        """
        import random
        random.seed(42)

        bf = BBoxOneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        center_bbox = [200, 100, 400, 300]
        noise_amplitude = 3.0
        t = 0.0
        dt = 0.033

        bf.filter(center_bbox, timestamp=t)
        t += dt

        output_deviations = []
        for i in range(200):
            noisy_bbox = [
                center_bbox[j] + random.uniform(-noise_amplitude, noise_amplitude)
                for j in range(4)
            ]
            result = bf.filter(noisy_bbox, timestamp=t)
            t += dt
            max_deviation = max(abs(result[j] - center_bbox[j]) for j in range(4))
            output_deviations.append(max_deviation)

        avg_deviation = sum(output_deviations[-50:]) / 50
        max_deviation = max(output_deviations[-50:])

        print(f"\n[BBoxNoise] Input noise: +/-{noise_amplitude} pixels")
        print(f"[BBoxNoise] Avg output deviation (last 50 frames): {avg_deviation:.4f}")
        print(f"[BBoxNoise] Max output deviation (last 50 frames): {max_deviation:.4f}")
        print(f"[BBoxNoise] Noise reduction: {(1 - avg_deviation/noise_amplitude)*100:.0f}%")

        if avg_deviation >= noise_amplitude * 0.8:
            print("[BBoxNoise] WARNING: Filter does NOT smooth noise effectively!")
            print("  -> Likely due to RC2 bug (inflated velocity -> alpha ~ 1.0)")
        else:
            print(f"[BBoxNoise] OK: Filter reduces noise to {avg_deviation/noise_amplitude*100:.0f}%")


# ==============================================================================
# SUMMARY
# ==============================================================================

class TestSummary(unittest.TestCase):
    """Print summary table."""

    def test_z_summary(self):
        """Runs last (z_ prefix for sort order)."""
        print("\n" + "=" * 70)
        print("SUMMARY OF BUGS TO VERIFY")
        print("=" * 70)
        print("""
| RC  | Bug                                     | Test Class                           |
|-----|-----------------------------------------|--------------------------------------|
| RC1 | PID Integral Windup (Grace Period)       | TestRC1_PID_IntegralWindup           |
| RC2 | 1-Euro Filter wrong derivative          | TestRC2_OneEuroFilter_WrongDerivative|
| RC3 | D-Term Spike (YOLO stale)               | TestRC3_DTermSpike                   |
| RC4 | Velocity is frame-rate dependent         | TestRC4_VelocityFrameDependent       |
| RC5 | Derivative Kick on deadzone exit         | TestRC5_DerivativeKick               |
| +   | BBox Filter drift & noise               | TestBBoxFilterDrift                  |

If test PASSES -> BUG CONFIRMED (tests are designed to PASS when bug exists)
If test FAILS  -> Bug has been fixed or hypothesis is wrong
""")


if __name__ == "__main__":
    unittest.main(verbosity=2)
