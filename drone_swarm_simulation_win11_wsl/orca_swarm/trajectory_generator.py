#!/usr/bin/env python3
"""
TRAJECTORY GENERATOR (B-SPLINE & MINIMUM SNAP)
Module tạo quỹ đạo mượt mà cấp C²/C³ cho Micro Quadrotors:
1. Minimum Snap Trajectory (Đa thức bậc 7 Mellinger & Kumar 2011):
   - Triệt tiêu hoàn toàn vận tốc, gia tốc, jerk ở 2 đầu mút: v(0)=0, v(T)=0, a(0)=0, a(T)=0, j(0)=0, j(T)=0.
   - Giúp biến thiên gia tốc và dòng điện động cơ quadrotor êm ái nhất.
2. B-Spline Trajectory (scipy.interpolate):
   - Quỹ đạo nội suy Spline liên tục C² qua chuỗi điểm nút (waypoints).

LƯU Ý THIẾT KẾ QUAN TRỌNG:
Quỹ đạo này chỉ đóng vai trò cung cấp Vector Vận Tốc Mong Muốn v_pref(t) và vị trí tham chiếu p_ref(t).
Thuật toán ORCA/RVO2 hoàn toàn có quyền uốn cong, dạt hướng và thay đổi vector vận tốc thực tế v_safe
trong không gian mở để né tránh va chạm, tuyệt đối KHÔNG kẹp cứng drone vào một đường ray cố định!
"""

import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


class MinimumSnapTrajectory:
    """
    Quỹ đạo Minimum Snap đa thức bậc 7 (7th-order polynomial) cho từng trục X, Y, Z.
    p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵ + c6*t⁶ + c7*t⁷
    với điều kiện biên nghỉ hoàn toàn ở cả 2 đầu:
    p(0) = p0, p(T) = p1
    v(0) = 0,  v(T) = 0
    a(0) = 0,  a(T) = 0
    j(0) = 0,  j(T) = 0
    """

    def __init__(self, p_start: Tuple[float, float, float], p_end: Tuple[float, float, float], duration: float):
        """
        :param p_start: Toạ độ xuất phát (x0, y0, z0)
        :param p_end: Toạ độ đích đến (x1, y1, z1)
        :param duration: Thời gian chuyển động T (giây)
        """
        self.p_start = np.array(p_start[:3], dtype=np.float64)
        self.p_end = np.array(p_end[:3], dtype=np.float64)
        self.duration = max(0.1, float(duration))
        self.dp = self.p_end - self.p_start

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Đánh giá trạng thái tại thời điểm t.
        :return: (vị trí p(t), vận tốc v(t), gia tốc a(t))
        """
        if t <= 0.0:
            return self.p_start.copy(), np.zeros(3), np.zeros(3)
        if t >= self.duration:
            return self.p_end.copy(), np.zeros(3), np.zeros(3)

        tau = t / self.duration  # Biến chuẩn hóa tau thuộc [0, 1]

        # Đa thức Minimum Snap bậc 7: s(tau) = 35*tau^4 - 84*tau^5 + 70*tau^6 - 20*tau^7
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau2 * tau2
        tau5 = tau3 * tau2
        tau6 = tau3 * tau3
        tau7 = tau4 * tau3

        s = 35.0 * tau4 - 84.0 * tau5 + 70.0 * tau6 - 20.0 * tau7
        ds_dtau = 140.0 * tau3 - 420.0 * tau4 + 420.0 * tau5 - 140.0 * tau6
        d2s_dtau2 = 420.0 * tau2 - 1680.0 * tau3 + 2100.0 * tau4 - 840.0 * tau5

        # Đạo hàm theo thời gian thực t = tau * T:
        pos = self.p_start + self.dp * s
        vel = self.dp * (ds_dtau / self.duration)
        acc = self.dp * (d2s_dtau2 / (self.duration * self.duration))

        return pos, vel, acc

    def get_preferred_velocity(
        self,
        t: float,
        p_current: np.ndarray,
        k_pos: float = 1.2,
        v_max: float = 1.8
    ) -> np.ndarray:
        """
        Tính toán Vector Vận Tốc Mong Muốn v_pref để nạp vào lõi ORCA:
        v_pref = v_ref(t) + k_pos * (p_ref(t) - p_current)
        Điều này cho phép drone tự do lệch khỏi quỹ đạo khi né va chạm,
        và từ từ quay trở lại bám đích mà không bị gò bó cưỡng bức.
        """
        p_ref, v_ref, _ = self.evaluate(t)
        err = p_ref - p_current[:3]
        v_pref = v_ref + k_pos * err

        # Khống chế trần vận tốc
        speed = np.linalg.norm(v_pref)
        if speed > v_max:
            v_pref = (v_pref / speed) * v_max

        return v_pref


class BSplineTrajectory:
    """
    Quỹ đạo nội suy B-Spline liên tục đạo hàm cấp 2 (C²) qua chuỗi toạ độ.
    """

    def __init__(self, waypoints: List[Tuple[float, float, float]], duration: float, degree: int = 3):
        """
        :param waypoints: Danh sách các điểm mốc [(x, y, z), ...] (tối thiểu 3 điểm)
        :param duration: Tổng thời gian đi qua toàn bộ quỹ đạo (giây)
        :param degree: Bậc spline (mặc định bậc 3 - Cubic B-Spline)
        """
        pts = np.array(waypoints, dtype=np.float64)
        if len(pts) < 2:
            raise ValueError("Cần tối thiểu 2 điểm để tạo B-Spline")

        self.duration = max(0.1, float(duration))
        self.p_start = pts[0].copy()
        self.p_end = pts[-1].copy()

        n_pts = len(pts)
        if n_pts == 2:
            # Nếu chỉ có 2 điểm, bổ sung điểm giữa để tạo spline bậc 3 mượt
            mid1 = pts[0] + (pts[1] - pts[0]) * 0.333
            mid2 = pts[0] + (pts[1] - pts[0]) * 0.667
            pts = np.vstack([pts[0], mid1, mid2, pts[1]])
            n_pts = 4

        actual_k = min(degree, n_pts - 1)
        t_steps = np.linspace(0.0, self.duration, n_pts)

        # Xây dựng B-Spline cho từng trục X, Y, Z với điều kiện biên kẹp chặt (clamped bc_type='clamped')
        self.splines = []
        self.spline_ders = []
        self.spline_der2s = []

        for axis in range(3):
            values = pts[:, axis]
            # make_interp_spline với bc_type=([(1, 0.0)], [(1, 0.0)]) bảo đảm vận tốc ở 2 đầu bằng 0
            spl = make_interp_spline(t_steps, values, k=actual_k, bc_type=([(1, 0.0)], [(1, 0.0)]))
            self.splines.append(spl)
            self.spline_ders.append(spl.derivative(1))
            self.spline_der2s.append(spl.derivative(2))

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Đánh giá trạng thái (p, v, a) tại thời điểm t.
        """
        t_clamped = max(0.0, min(self.duration, float(t)))
        pos = np.array([float(self.splines[i](t_clamped)) for i in range(3)], dtype=np.float64)
        vel = np.array([float(self.spline_ders[i](t_clamped)) for i in range(3)], dtype=np.float64)
        acc = np.array([float(self.spline_der2s[i](t_clamped)) for i in range(3)], dtype=np.float64)
        return pos, vel, acc

    def get_preferred_velocity(
        self,
        t: float,
        p_current: np.ndarray,
        k_pos: float = 1.2,
        v_max: float = 1.8
    ) -> np.ndarray:
        """
        Tính toán Vector Vận Tốc Mong Muốn v_pref từ B-Spline:
        v_pref = v_ref(t) + k_pos * (p_ref(t) - p_current)
        """
        p_ref, v_ref, _ = self.evaluate(t)
        err = p_ref - p_current[:3]
        v_pref = v_ref + k_pos * err

        speed = np.linalg.norm(v_pref)
        if speed > v_max:
            v_pref = (v_pref / speed) * v_max

        return v_pref


class TrajectoryGenerator:
    """
    Lớp điều phối tạo quỹ đạo bầy đàn (hỗ trợ cả Minimum Snap và B-Spline).
    """

    @staticmethod
    def create_swarm_trajectories(
        current_poses: Dict[int, Tuple[float, ...]],
        assigned_targets: Dict[int, Tuple[float, ...]],
        duration: float,
        traj_type: str = "min_snap"
    ) -> Dict[int, Any]:
        """
        Tạo bộ quỹ đạo song song cho toàn bộ các drone trong bầy.

        :param current_poses: Dict {drone_id: (x, y, z, ...)}
        :param assigned_targets: Dict {drone_id: (gx, gy, gz, ...)}
        :param duration: Thời gian chuyển động T (giây)
        :param traj_type: "min_snap" hoặc "bspline"
        :return: Dict {drone_id: TrajectoryObject}
        """
        trajectories = {}
        for d_id, p_start in current_poses.items():
            if d_id not in assigned_targets:
                continue
            p_end = assigned_targets[d_id]
            p0 = (float(p_start[0]), float(p_start[1]), float(p_start[2]) if len(p_start) > 2 else 1.3)
            p1 = (float(p_end[0]), float(p_end[1]), float(p_end[2]) if len(p_end) > 2 else 1.3)

            if traj_type == "bspline":
                trajectories[d_id] = BSplineTrajectory([p0, p1], duration=duration)
            else:
                trajectories[d_id] = MinimumSnapTrajectory(p0, p1, duration=duration)

        return trajectories
