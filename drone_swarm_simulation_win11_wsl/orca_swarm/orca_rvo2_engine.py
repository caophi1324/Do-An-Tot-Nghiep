#!/usr/bin/env python3
"""
ORCA / RVO2 COLLISION AVOIDANCE ENGINE
Lõi tính toán né tránh va chạm tương hỗ không gian vận tốc (Optimal Reciprocal Collision Avoidance):
- Triển khai thuật toán ORCA/RVO2 nguyên bản (J. van den Berg et al. 2011).
- Nguyên lý chia sẻ trách nhiệm 50-50 tương hỗ (Reciprocal Avoidance).
- Nón vận tốc va chạm (Velocity Obstacle - VO) với chân trời thời gian tau.
- Bộ giải Quy hoạch Tuyến tính chuẩn mực (linearProgram1, linearProgram2, linearProgram3) O(M) độ phức tạp.
- Tuyệt đối KHÔNG bắt buộc drone đi theo đường ray cứng; drone tự do lách hướng để né tránh va chạm.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class ORCAConfig:
    """Tham số cấu hình thuật toán ORCA/RVO2"""
    time_horizon: float = 2.0       # (s) Chân trời thời gian dự đoán va chạm (tau)
    agent_radius: float = 0.40      # (m) Bán kính bảo vệ của mỗi drone (drone micro 22cm + biên an toàn)
    max_speed: float = 2.4          # (m/s) Vận tốc tối đa cho phép của drone
    time_step: float = 0.1          # (s) Chu kỳ điều khiển vi bước (dt @ 10Hz)
    safety_margin: float = 0.10     # (m) Biên độ đệm an toàn dự phòng
    z_anisotropy: float = 1.2       # Hệ số dãn cách trục Z (chống downwash)


class ORCAAgent:
    """Đại diện cho một drone trong không gian tính toán ORCA."""

    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        pref_velocity: np.ndarray,
        radius: float = 0.40,
        max_speed: float = 2.4
    ):
        self.id = agent_id
        self.position = np.array(position[:3], dtype=np.float64)
        self.velocity = np.array(velocity[:3], dtype=np.float64)
        self.pref_velocity = np.array(pref_velocity[:3], dtype=np.float64)
        self.radius = radius
        self.max_speed = max_speed


class ORCALine:
    """
    Ràng buộc nửa mặt phẳng ORCA trong không gian 2D:
    det(direction, point - v) <= 0
    trong đó direction là véc-tơ đơn vị chỉ hướng của đường thẳng.
    """

    def __init__(self, point: np.ndarray, direction: np.ndarray):
        self.point = np.array(point[:2], dtype=np.float64)
        norm = np.linalg.norm(direction[:2])
        if norm > 1e-8:
            self.direction = np.array(direction[:2], dtype=np.float64) / norm
        else:
            self.direction = np.array([1.0, 0.0], dtype=np.float64)


def det_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Định thức 2D det([v1, v2]) = v1.x * v2.y - v1.y * v2.x"""
    return float(v1[0] * v2[1] - v1[1] * v2[0])


class ORCA2DEngine:
    """
    Bộ giải thuật toán ORCA-2D (Optimal Reciprocal Collision Avoidance).
    Giải bài toán tìm vận tốc an toàn 2D trong mặt phẳng ngang XY,
    kết hợp với phân tầng cao độ Z mượt mà.
    """

    def __init__(self, config: Optional[ORCAConfig] = None):
        self.config = config or ORCAConfig()

    def compute_orca_line(
        self,
        agent: ORCAAgent,
        neighbor: ORCAAgent
    ) -> ORCALine:
        """
        Tính toán đường ràng buộc nửa mặt phẳng ORCA giữa 2 drone (chuẩn RVO2).
        Chia sẻ 50% trách nhiệm né tránh tương hỗ.
        """
        rel_pos = neighbor.position[:2] - agent.position[:2]
        rel_vel = agent.velocity[:2] - neighbor.velocity[:2]
        dist_sq = float(np.dot(rel_pos, rel_pos))
        combined_radius = agent.radius + neighbor.radius + self.config.safety_margin
        combined_radius_sq = combined_radius * combined_radius

        inv_tau = 1.0 / self.config.time_horizon
        line = ORCALine(np.zeros(2), np.zeros(2))

        if dist_sq > combined_radius_sq:
            # Hai drone chưa va chạm: tính toán nón va chạm VO cắt cụt
            w = rel_vel - inv_tau * rel_pos
            w_len_sq = float(np.dot(w, w))
            dot1 = float(np.dot(w, rel_pos))

            if dot1 < 0.0 and (dot1 * dot1) > (combined_radius_sq * w_len_sq):
                # Chiếu vào đường tròn cắt cụt (Cut-off circle)
                w_len = math.sqrt(w_len_sq)
                unit_w = w / w_len
                line.direction = np.array([unit_w[1], -unit_w[0]], dtype=np.float64)
                u = (combined_radius * inv_tau - w_len) * unit_w
            else:
                # Chiếu vào hai cạnh bên của nón va chạm (Legs)
                leg = math.sqrt(dist_sq - combined_radius_sq)
                if det_2d(rel_pos, w) > 0.0:
                    # Cạnh bên trái
                    line.direction = np.array([
                        rel_pos[0] * leg - rel_pos[1] * combined_radius,
                        rel_pos[0] * combined_radius + rel_pos[1] * leg
                    ], dtype=np.float64) / dist_sq
                else:
                    # Cạnh bên phải
                    line.direction = -np.array([
                        rel_pos[0] * leg + rel_pos[1] * combined_radius,
                        -rel_pos[0] * combined_radius + rel_pos[1] * leg
                    ], dtype=np.float64) / dist_sq

                dot2 = float(np.dot(rel_vel, line.direction))
                u = dot2 * line.direction - rel_vel
        else:
            # Đang chồng lấn / quá gần: dùng chu kỳ vi bước dt để né tức thì
            inv_dt = 1.0 / self.config.time_step
            w = rel_vel - inv_dt * rel_pos
            w_len = np.linalg.norm(w)
            if w_len > 1e-8:
                unit_w = w / w_len
            else:
                unit_w = np.array([1.0, 0.0], dtype=np.float64)
            line.direction = np.array([unit_w[1], -unit_w[0]], dtype=np.float64)
            u = (combined_radius * inv_dt - w_len) * unit_w

        # Chia sẻ 50% trách nhiệm né tránh tương hỗ
        line.point = agent.velocity[:2] + 0.5 * u
        return line

    def linear_program_1(
        self,
        lines: List[ORCALine],
        line_no: int,
        radius: float,
        opt_vel: np.ndarray,
        direction_opt: bool
    ) -> Tuple[bool, np.ndarray]:
        """Quy hoạch tuyến tính 1D trên đường biên line_no (RVO2 standard)."""
        dot = float(np.dot(lines[line_no].point, lines[line_no].direction))
        discriminant = dot * dot + radius * radius - float(np.dot(lines[line_no].point, lines[line_no].point))

        if discriminant < 0.0:
            return False, np.zeros(2)

        sqrt_disc = math.sqrt(max(0.0, discriminant))
        t_left = -dot - sqrt_disc
        t_right = -dot + sqrt_disc

        for i in range(line_no):
            denom = det_2d(lines[line_no].direction, lines[i].direction)
            numer = det_2d(lines[i].direction, lines[line_no].point - lines[i].point)

            if abs(denom) <= 1e-8:
                if numer < 0.0:
                    return False, np.zeros(2)
                continue

            t = numer / denom
            if denom >= 0.0:
                t_right = min(t_right, t)
            else:
                t_left = max(t_left, t)

            if t_left > t_right:
                return False, np.zeros(2)

        if direction_opt:
            if float(np.dot(opt_vel, lines[line_no].direction)) > 0.0:
                result = lines[line_no].point + t_right * lines[line_no].direction
            else:
                result = lines[line_no].point + t_left * lines[line_no].direction
        else:
            t = float(np.dot(lines[line_no].direction, opt_vel - lines[line_no].point))
            if t < t_left:
                result = lines[line_no].point + t_left * lines[line_no].direction
            elif t > t_right:
                result = lines[line_no].point + t_right * lines[line_no].direction
            else:
                result = lines[line_no].point + t * lines[line_no].direction

        return True, result

    def linear_program_2(
        self,
        lines: List[ORCALine],
        radius: float,
        opt_vel: np.ndarray,
        direction_opt: bool
    ) -> Tuple[int, np.ndarray]:
        """Quy hoạch tuyến tính 2D gia tăng O(M) (RVO2 standard)."""
        if direction_opt:
            result = opt_vel * radius
        elif float(np.dot(opt_vel, opt_vel)) > radius * radius:
            speed = np.linalg.norm(opt_vel)
            result = (opt_vel / speed) * radius if speed > 1e-8 else np.zeros(2)
        else:
            result = opt_vel.copy()

        for i in range(len(lines)):
            if det_2d(lines[i].direction, lines[i].point - result) > 0.0:
                success, new_result = self.linear_program_1(lines, i, radius, opt_vel, direction_opt)
                if not success:
                    return i, result
                result = new_result

        return len(lines), result

    def linear_program_3(
        self,
        lines: List[ORCALine],
        num_obst_lines: int,
        begin_line: int,
        radius: float,
        result: np.ndarray
    ) -> np.ndarray:
        """Cơ chế nới lỏng 3D LP khi tập nghiệm rỗng (Relaxation fallback)."""
        distance = 0.0
        cur_res = result.copy()

        for i in range(begin_line, len(lines)):
            if det_2d(lines[i].direction, lines[i].point - cur_res) > distance:
                proj_lines = list(lines[:num_obst_lines])
                for j in range(num_obst_lines, i):
                    det = det_2d(lines[i].direction, lines[j].direction)
                    line_j = ORCALine(np.zeros(2), np.zeros(2))
                    if abs(det) <= 1e-8:
                        if float(np.dot(lines[i].direction, lines[j].direction)) > 0.0:
                            continue
                        else:
                            line_j.point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        line_j.point = lines[i].point + (det_2d(lines[j].direction, lines[i].point - lines[j].point) / det) * lines[i].direction

                    diff = lines[j].direction - lines[i].direction
                    norm_diff = np.linalg.norm(diff)
                    line_j.direction = diff / norm_diff if norm_diff > 1e-8 else np.array([1.0, 0.0])
                    proj_lines.append(line_j)

                temp_result = cur_res.copy()
                opt_dir = np.array([-lines[i].direction[1], lines[i].direction[0]], dtype=np.float64)
                line_fail, lp_res = self.linear_program_2(proj_lines, radius, opt_dir, True)
                if line_fail == len(proj_lines):
                    cur_res = lp_res
                else:
                    cur_res = temp_result
                distance = det_2d(lines[i].direction, lines[i].point - cur_res)

        return cur_res

    def compute_agent_velocity(
        self,
        agent: ORCAAgent,
        neighbors: List[ORCAAgent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Tính toán vận tốc an toàn 3D cho 1 drone:
        - Bình diện ngang XY giải bằng ORCA-2D (RVO2) có bổ sung Right-Hand Aviation Bias.
        - Trục thẳng đứng Z giải bằng bám theo cao độ mong muốn có khống chế gia tốc.
        """
        orca_lines = []
        min_dist = float('inf')

        # RORCAL 2026 / Aviation Right-Hand Rule Bias:
        # Bổ sung vector lệch hướng tiếp tuyến rẽ phải vào v_opt khi có nguy cơ tiếp cận đối đầu
        v_opt = agent.pref_velocity[:2].copy()

        for other in neighbors:
            if other.id == agent.id:
                continue

            rel_pos = other.position[:2] - agent.position[:2]
            dist_2d = float(np.linalg.norm(rel_pos))
            dist_3d = float(np.linalg.norm(agent.position - other.position))

            if dist_3d < min_dist:
                min_dist = dist_3d

            # Quy tắc rẽ phải hàng không (Right-Hand Aviation Bias):
            # Khi cự ly 2D < 1.8m và agent đang có xu hướng tiến lại gần đối phương:
            if 1e-4 < dist_2d < 1.8:
                approach = float(np.dot(agent.pref_velocity[:2], rel_pos))
                if approach > -0.05:
                    # Vector pháp tuyến vuông góc hướng sang bên phải của agent (Clockwise 90 deg): u_right = [ry, -rx] / |r|
                    u_right = np.array([rel_pos[1], -rel_pos[0]], dtype=np.float64) / dist_2d
                    # Cường độ né tăng dần khi cự ly thu hẹp từ 1.8m xuống 0.0m (tối đa 0.6 m/s)
                    bias_weight = max(0.0, min(1.0, (1.8 - dist_2d) / 1.8))
                    v_opt += 0.60 * bias_weight * u_right

            line = self.compute_orca_line(agent, other)
            orca_lines.append(line)

        # Giải vận tốc tối ưu 2D trong mặt phẳng XY
        line_fail, v_safe_xy = self.linear_program_2(orca_lines, agent.max_speed, v_opt, False)

        if line_fail < len(orca_lines):
            v_safe_xy = self.linear_program_3(orca_lines, 0, line_fail, agent.max_speed, v_safe_xy)

        # Trục Z: bám vận tốc mong muốn trục Z
        v_safe_z = float(agent.pref_velocity[2])
        # Khống chế tốc độ leo/hạ cao độ an toàn (tối đa 0.8 m/s)
        v_safe_z = max(-0.8, min(0.8, v_safe_z))

        v_safe_3d = np.array([v_safe_xy[0], v_safe_xy[1], v_safe_z], dtype=np.float64)

        stats = {
            "min_distance_m": float(min_dist) if min_dist != float('inf') else 99.0,
            "num_constraints": len(orca_lines),
            "speed_xy": float(np.linalg.norm(v_safe_xy)),
            "speed_z": float(abs(v_safe_z)),
        }

        return v_safe_3d, stats


class ORCA3DEngine:
    """
    Bộ giải ORCA trong không gian 3D toàn phần (3D Half-Spaces).
    Áp dụng khi bầy drone bay đội hình phức tạp với biến thiên cao độ lớn.
    """

    def __init__(self, config: Optional[ORCAConfig] = None):
        self.config = config or ORCAConfig()

    def compute_agent_velocity(
        self,
        agent: ORCAAgent,
        neighbors: List[ORCAAgent]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Tính toán vận tốc an toàn 3D đầy đủ.
        """
        tau = self.config.time_horizon
        dt = self.config.time_step
        r_sum = agent.radius * 2.0 + self.config.safety_margin
        v_pref = agent.pref_velocity.copy()

        planes_point = []
        planes_normal = []
        min_dist = float('inf')

        for other in neighbors:
            if other.id == agent.id:
                continue

            rel_pos = other.position - agent.position
            # Tăng trọng số an toàn trục Z để chống downwash
            rel_pos_aniso = rel_pos.copy()
            rel_pos_aniso[2] /= self.config.z_anisotropy

            dist = np.linalg.norm(rel_pos_aniso)
            if dist < min_dist:
                min_dist = dist

            rel_vel = agent.velocity - other.velocity

            if dist > r_sum:
                w = rel_vel - (rel_pos_aniso / tau)
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-6:
                    n = w / w_norm
                    u = (r_sum / tau - w_norm) * n
                else:
                    n = np.array([1.0, 0.0, 0.0])
                    u = np.zeros(3)
            else:
                w = rel_vel - (rel_pos_aniso / dt)
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-6:
                    n = w / w_norm
                    u = (r_sum / dt - w_norm) * n
                else:
                    n = np.array([1.0, 0.0, 0.0])
                    u = np.zeros(3)

            planes_point.append(agent.velocity + 0.5 * u)
            planes_normal.append(n)

        # Giải tối ưu 3D bằng chiếu chiếu trực giao lặp (Projected Gradient)
        v_safe = v_pref.copy()
        speed = np.linalg.norm(v_safe)
        if speed > agent.max_speed:
            v_safe = (v_safe / speed) * agent.max_speed

        # Lặp chiếu các nửa không gian (Half-space projections)
        for _ in range(5):
            any_violation = False
            for pt, n in zip(planes_point, planes_normal):
                val = float(np.dot(v_safe - pt, n))
                if val < 0.0:
                    v_safe -= val * n  # Đẩy ra mặt phẳng an toàn
                    any_violation = True
            if not any_violation:
                break

        speed = np.linalg.norm(v_safe)
        if speed > agent.max_speed:
            v_safe = (v_safe / speed) * agent.max_speed

        stats = {
            "min_distance_m": float(min_dist) if min_dist != float('inf') else 99.0,
            "speed": float(speed)
        }
        return v_safe, stats
