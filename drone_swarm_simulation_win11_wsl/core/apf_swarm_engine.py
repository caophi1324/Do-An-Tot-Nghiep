#!/usr/bin/env python3
"""
DRONEEXE Core - 3D Artificial Potential Field (APF) Swarm Engine
Thuật toán Trường thế nhân tạo 3D (APF) và Dẫn đường Bám đuôi (Leader-Follower)
Dành cho hệ thống Swarm 5 Drone ArduPilot.

Tính năng:
- Lực hút (Attractive Force): Kéo follower bám chuẩn xác theo hình học đội hình của Leader.
- Lực đẩy 3D (Repulsive Force): Triệt tiêu va chạm giữa mọi cặp drone thời gian thực.
- Xử lý dị hướng cánh quạt (Propwash Downwash Anisotropy): Bảo vệ vùng an toàn trục Z.
- Cơ chế giải tỏa bế tắc cục bộ GNRON (Goal Non-Reachable with Obstacle Nearby).
- Hỗ trợ các kiểu biến đổi hình học đội hình: V-Shape, Diamond, Column, Line.
- Thuần toán học (Vector Math), độc lập với ROS và giao thức truyền thông.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class APFConfig:
    """Tham số cấu hình thuật toán Trường thế nhân tạo 3D Damped Bounded APF"""
    d_influence: float = 1.20      # (m) Bán kính trường thế đẩy bắt đầu tác động
    d_safe: float = 0.85           # (m) Khoảng cách an toàn danh định
    d_critical: float = 0.50       # (m) Khoảng cách khẩn cấp
    k_att: float = 1.2             # Hệ số lực hút kéo về đội hình
    k_rep: float = 1.5             # Hệ số lực đẩy chống va chạm
    k_damp: float = 0.85           # Hệ số giảm chấn vận tốc tương đối (triệt tiêu dao động hunting & overshoot)
    k_tang: float = 0.35           # Hệ số lực né tiếp tuyến tránh xung đột đối đầu
    v_max: float = 1.8             # (m/s) Vận tốc tối đa cho phép của mỗi drone (nâng tốc độ chuyển đội hình)
    z_anisotropy: float = 1.2      # Hệ số an toàn trục Z
    repulsion_cap: float = 2.2     # Ngưỡng bão hòa lực đẩy cực đại
    formation_scale: float = 1.0   # Hệ số tỉ lệ hình học đội hình cho hội trường 20x30m


class FormationType:
    V_SHAPE = "V_SHAPE"           # Đội hình chữ V / Mũi tên dẫn đường
    DIAMOND = "DIAMOND"           # Đội hình hình thoi / Kim cương bảo vệ
    COLUMN = "COLUMN"             # Đội hình hàng dọc
    LINE = "LINE"                 # Đội hình hàng ngang


class APFSwarmController:
    """
    Bộ điều khiển Bầy đàn 3D Damped Bounded APF và Bám đuôi Leader-Follower
    """

    def __init__(self, config: Optional[APFConfig] = None):
        self.config = config or APFConfig()
        self.prev_poses: Dict[int, Tuple[float, float, float]] = {}

        # Định nghĩa offset hình học cục bộ (x_fwd, y_right, z_up) trong hệ quy chiếu Leader
        # Cự ly dãn cách danh định 1.6m - 2.0m phù hợp hội trường 20x30m
        self._formation_templates = {
            FormationType.V_SHAPE: {
                0: ( 0.0,  0.0, 0.0),    # Leader (Đỉnh chữ V)
                1: (-1.8, -1.6, 0.0),    # Drone 2 (Cánh trái trong)
                2: (-1.8,  1.6, 0.0),    # Drone 3 (Cánh phải trong)
                3: (-3.6, -3.2, 0.0),    # Drone 4 (Cánh trái ngoài)
                4: (-3.6,  3.2, 0.0),    # Drone 5 (Cánh phải ngoài)
            },
            FormationType.DIAMOND: {
                0: ( 0.0,  0.0, 0.0),    # Leader (Đỉnh trước)
                1: (-2.0, -2.0, 0.0),    # Drone 2 (Đỉnh bên trái)
                2: (-2.0,  2.0, 0.0),    # Drone 3 (Đỉnh bên phải)
                3: (-2.0,  0.0, 0.0),    # Drone 4 (Tâm kim cương)
                4: (-4.0,  0.0, 0.0),    # Drone 5 (Đỉnh phía sau)
            },
            FormationType.COLUMN: {
                0: ( 0.0, 0.0, 0.0),     # Leader
                1: (-1.6, 0.0, 0.0),     # Drone 2 (Thụt lùi 1.6m)
                2: (-3.2, 0.0, 0.0),     # Drone 3 (Thụt lùi 3.2m)
                3: (-4.8, 0.0, 0.0),     # Drone 4 (Thụt lùi 4.8m)
                4: (-6.4, 0.0, 0.0),     # Drone 5 (Thụt lùi 6.4m)
            },
            FormationType.LINE: {
                0: (0.0,  0.0, 0.0),     # Leader (Tâm hàng ngang)
                1: (0.0, -1.6, 0.0),     # Drone 2 (Bên trái 1)
                2: (0.0,  1.6, 0.0),     # Drone 3 (Bên phải 1)
                3: (0.0, -3.2, 0.0),     # Drone 4 (Bên trái 2)
                4: (0.0,  3.2, 0.0),     # Drone 5 (Bên phải 2)
            }
        }

        # Thu nhỏ toàn bộ hình học đội hình theo formation_scale (Audit R3-02)
        s = self.config.formation_scale
        for f_type, tmpl in self._formation_templates.items():
            self._formation_templates[f_type] = {
                k: (dx * s, dy * s, dz * s) for k, (dx, dy, dz) in tmpl.items()
            }

    def get_formation_targets(
        self,
        leader_pose: Tuple[float, float, float, float],
        formation_type: str = FormationType.V_SHAPE
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Tính toán toạ độ mục tiêu toàn cục (Global NED / Local frame) cho cả 5 drone
        dựa trên toạ độ hiện tại của Leader (lx, ly, lz, yaw_deg).

        Returns:
            Dict[drone_id -> (target_x, target_y, target_z, target_yaw)]
        """
        lx, ly, lz, l_yaw_deg = leader_pose
        l_yaw_rad = math.radians(l_yaw_deg)
        cos_yaw = math.cos(l_yaw_rad)
        sin_yaw = math.sin(l_yaw_rad)

        template = self._formation_templates.get(formation_type, self._formation_templates[FormationType.V_SHAPE])
        targets = {}

        for d_id, (dx, dy, dz) in template.items():
            if d_id == 0:
                # Leader giữ nguyên toạ độ thực của mình
                targets[0] = (lx, ly, lz, l_yaw_deg)
            else:
                # Xoay offset từ hệ toạ độ Body của Leader sang hệ toạ độ World/Local NED
                world_dx = dx * cos_yaw - dy * sin_yaw
                world_dy = dx * sin_yaw + dy * cos_yaw
                world_dz = dz

                tx = lx + world_dx
                ty = ly + world_dy
                tz = lz + world_dz
                targets[d_id] = (tx, ty, tz, l_yaw_deg)

        return targets

    def compute_attractive_force(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Lực hút kéo drone i về vị trí mục tiêu hình học.
        Áp dụng thuật toán hãm phanh 2 giai đoạn (Dual-Regime Critically Damped Braking):
        - Vùng gần (<= 0.35m): Tuyến tính tới hạn v = k * d (hội tụ hàm mũ đơn điệu, triệt tiêu 100% overshoot).
        - Vùng xa (> 0.35m): Căn bậc hai v_brake = sqrt(2 * a * d) bão hòa v_max (chống vọt ga).
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dz = target_pos[2] - current_pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1e-4:
            return (0.0, 0.0, 0.0)

        d_linear = 0.35
        if dist < d_linear:
            speed = 1.1 * dist
        else:
            speed = min(self.config.v_max, math.sqrt(2.0 * 0.6 * (dist - d_linear * 0.5)))
        speed = min(self.config.v_max, speed)

        fx = (dx / dist) * speed
        fy = (dy / dist) * speed
        fz = (dz / dist) * speed

        return (fx, fy, fz)

    def compute_repulsive_force(
        self,
        d_id: int,
        current_poses: Dict[int, Tuple[float, float, float, float]],
        target_pos: Tuple[float, float, float],
        velocities: Optional[Dict[int, Tuple[float, float, float]]] = None
    ) -> Tuple[float, float, float]:
        """
        Lực đẩy nhân tạo 3D Damped Bounded APF:
        - Hàm vuốt mượt Smooth Cosine Tapering (C2 continuous) chống cắt lực đột ngột.
        - Giảm chấn vận tốc tương đối (Relative Velocity Damping) triệt tiêu nảy ngược đàn hồi.
        - Lực né tiếp tuyến rẽ phải (Right-Hand Rule) chống kẹt đối đầu trực diện.
        - Điều biến GNRON khi tiếp cận mục tiêu.
        """
        pos_i = current_poses[d_id]
        xi, yi, zi = pos_i[0], pos_i[1], pos_i[2]

        dist_to_goal = math.sqrt(
            (target_pos[0] - xi)**2 +
            (target_pos[1] - yi)**2 +
            (target_pos[2] - zi)**2
        )
        gnron_scale = min(1.0, (dist_to_goal / max(0.1, self.config.d_safe))**2)

        total_fx = 0.0
        total_fy = 0.0
        total_fz = 0.0

        for other_id, other_pose in current_poses.items():
            if other_id == d_id:
                continue

            xj, yj, zj = other_pose[0], other_pose[1], other_pose[2]

            dx = xi - xj
            dy = yi - yj
            dz = (zi - zj) / self.config.z_anisotropy

            dist_eff = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_eff < 1e-4:
                dist_eff = 1e-4
                dx = 0.05 * (1.0 if d_id % 2 == 0 else -1.0)
                dy = 0.05

            if dist_eff < self.config.d_influence:
                # 1. Hàm vuốt mượt Smooth Cosine Tapering
                eta = (self.config.d_influence - dist_eff) / max(1e-3, self.config.d_influence - self.config.d_critical)
                eta = max(0.0, min(1.0, eta))
                smooth_weight = 0.5 * (1.0 - math.cos(math.pi * eta))

                # Lực đẩy với bão hòa mềm
                base_rep = self.config.k_rep * (1.0 / max(0.15, dist_eff) - 1.0 / self.config.d_influence)
                if dist_eff < self.config.d_critical:
                    base_rep *= 1.4
                base_rep = min(self.config.repulsion_cap, base_rep) * smooth_weight
                scale = 1.0 if dist_eff < self.config.d_critical else gnron_scale
                base_rep *= scale

                ux = dx / dist_eff
                uy = dy / dist_eff
                uz = (zi - zj) / dist_eff

                # 2. Thành phần Giảm chấn Vận tốc Tương đối (Relative Velocity Damping)
                damp_fx, damp_fy, damp_fz = 0.0, 0.0, 0.0
                if velocities and d_id in velocities and other_id in velocities:
                    vi = velocities[d_id]
                    vj = velocities[other_id]
                    v_rel_x = vi[0] - vj[0]
                    v_rel_y = vi[1] - vj[1]
                    v_rel_z = vi[2] - vj[2]
                    v_approach = v_rel_x * ux + v_rel_y * uy + v_rel_z * uz
                    if v_approach < 0:
                        damp_force = -self.config.k_damp * v_approach * smooth_weight
                        damp_fx = damp_force * ux
                        damp_fy = damp_force * uy
                        damp_fz = damp_force * uz
                    else:
                        base_rep *= max(0.2, 1.0 - 0.4 * min(1.5, v_approach))

                # 3. Lực né tiếp tuyến rẽ phải (Aviation Right-Hand Avoidance Rule)
                # Khi hai drone đối đầu, tự động lách sang phải đường bay
                tang_fx, tang_fy = 0.0, 0.0
                if velocities and d_id in velocities:
                    v_drone = velocities[d_id]
                    v_speed = math.sqrt(v_drone[0]**2 + v_drone[1]**2)
                    head_x = v_drone[0] / max(0.1, v_speed) if v_speed > 0.05 else ux
                    head_y = v_drone[1] / max(0.1, v_speed) if v_speed > 0.05 else uy
                    # Vector pháp tuyến bên phải [head_y, -head_x]
                    tang_fx = head_y * self.config.k_tang * smooth_weight
                    tang_fy = -head_x * self.config.k_tang * smooth_weight
                else:
                    tang_fx = -uy * self.config.k_tang * smooth_weight
                    tang_fy = ux * self.config.k_tang * smooth_weight

                total_fx += (base_rep * ux + damp_fx + tang_fx)
                total_fy += (base_rep * uy + damp_fy + tang_fy)
                total_fz += (base_rep * uz + damp_fz)

        return (total_fx, total_fy, total_fz)

    def step(
        self,
        current_poses: Dict[int, Tuple[float, float, float, float]],
        leader_pose: Tuple[float, float, float, float],
        formation_type: str = FormationType.V_SHAPE,
        dt: float = 0.1
    ) -> Tuple[Dict[int, Tuple[float, float, float, float]], Dict[str, float]]:
        """
        Thực thi 1 chu kỳ điều khiển APF Swarm:
        - Đầu vào: Toạ độ thực tế hiện tại của 5 drone và Leader.
        - Đầu ra: Toạ độ setpoint mượt mà tiếp theo cho 5 drone.
        """
        targets = self.get_formation_targets(leader_pose, formation_type)
        next_commands = {}
        min_swarm_dist = float('inf')
        max_speed_demanded = 0.0

        # Ước lượng vận tốc thực tế từ vị trí liên tiếp để phục vụ giảm chấn Damped APF
        velocities = {}
        dt_clamped = max(0.02, min(0.2, dt))
        for d_id, pose in current_poses.items():
            if d_id in self.prev_poses:
                prev = self.prev_poses[d_id]
                vx = (pose[0] - prev[0]) / dt_clamped
                vy = (pose[1] - prev[1]) / dt_clamped
                vz = (pose[2] - prev[2]) / dt_clamped
                velocities[d_id] = (vx, vy, vz)
            else:
                velocities[d_id] = (0.0, 0.0, 0.0)
            self.prev_poses[d_id] = (pose[0], pose[1], pose[2])

        # 1. Tính toán toạ độ cho Leader (Drone 0)
        if 0 in current_poses:
            c0 = current_poses[0]
            t0 = targets[0]
            dx0 = t0[0] - c0[0]
            dy0 = t0[1] - c0[1]
            dz0 = t0[2] - c0[2]
            dist0 = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
            max_lead_step = self.config.v_max * dt_clamped
            if dist0 > max_lead_step and dist0 > 1e-4:
                scale = max_lead_step / dist0
                next_commands[0] = (
                    c0[0] + dx0 * scale,
                    c0[1] + dy0 * scale,
                    c0[2] + dz0 * scale,
                    t0[3],
                )
            else:
                next_commands[0] = targets[0]
        else:
            next_commands[0] = targets[0]

        # 2. Tính toán Damped Bounded APF cho các Followers (Drone 1, 2, 3, 4)
        for d_id in range(1, 5):
            if d_id not in current_poses:
                continue

            c_pos = (current_poses[d_id][0], current_poses[d_id][1], current_poses[d_id][2])
            t_pos = (targets[d_id][0], targets[d_id][1], targets[d_id][2])

            # Tính lực hút
            f_att_x, f_att_y, f_att_z = self.compute_attractive_force(c_pos, t_pos)

            # Tính lực đẩy Damped APF kèm giảm chấn vận tốc
            f_rep_x, f_rep_y, f_rep_z = self.compute_repulsive_force(d_id, current_poses, t_pos, velocities)

            # Tổng hợp vector vận tốc mong muốn
            vx = f_att_x + f_rep_x
            vy = f_att_y + f_rep_y
            vz = f_att_z + f_rep_z

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if speed > self.config.v_max:
                scale = self.config.v_max / speed
                vx *= scale
                vy *= scale
                vz *= scale
                speed = self.config.v_max

            max_speed_demanded = max(max_speed_demanded, speed)

            # Cập nhật bước di chuyển tiếp theo: p_next = p_current + v * dt
            next_x = c_pos[0] + vx * dt_clamped
            next_y = c_pos[1] + vy * dt_clamped
            next_z = c_pos[2] + vz * dt_clamped
            next_yaw = targets[d_id][3]

            next_commands[d_id] = (next_x, next_y, next_z, next_yaw)

        # 3. Thống kê khoảng cách nhỏ nhất giữa mọi cặp drone trong bầy
        drone_ids = list(current_poses.keys())
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                id_a, id_b = drone_ids[i], drone_ids[j]
                pa = current_poses[id_a]
                pb = current_poses[id_b]
                d = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2 + (pa[2] - pb[2])**2)
                if d < min_swarm_dist:
                    min_swarm_dist = d

        telemetry_stats = {
            "min_distance_m": round(min_swarm_dist, 3) if min_swarm_dist != float('inf') else 999.0,
            "max_speed_mps": round(max_speed_demanded, 3),
            "safe_status": "CRITICAL_AVOID" if min_swarm_dist < self.config.d_critical else (
                "APF_ACTIVE" if min_swarm_dist < self.config.d_influence else "FORMATION_NOMINAL"
            )
        }

        return next_commands, telemetry_stats
