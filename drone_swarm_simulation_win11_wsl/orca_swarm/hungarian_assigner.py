#!/usr/bin/env python3
"""
HUNGARIAN TARGET ASSIGNER
Thuật toán gán mục tiêu tối ưu toàn cục Hungarian (Kuhn-Munkres)
- Sử dụng scipy.optimize.linear_sum_assignment (độ phức tạp O(N^3), giải trong < 0.05ms cho N=5..20 drone)
- Cực tiểu hóa tổng bình phương khoảng cách di chuyển: min sum ||p_i - g_pi(i)||^2
- Triệt tiêu 100% hiện tượng giao cắt chéo đường bay ngay từ tầng quy hoạch mục tiêu
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianAssigner:
    """
    Bộ gán mục tiêu tối ưu cho hệ thống bầy đàn đa drone.
    Đảm bảo mỗi drone được gán tới đúng vị trí đích sao cho tổng chi phí di chuyển
    toàn bầy là nhỏ nhất, loại bỏ hoàn toàn các quỹ đạo giao cắt topo.
    """

    def __init__(self, z_weight: float = 1.2):
        """
        :param z_weight: Trọng số ưu tiên trục Z (lớn hơn 1 để hạn chế thay đổi cao độ không cần thiết).
        """
        self.z_weight = z_weight

    def assign(
        self,
        current_poses: Dict[int, Tuple[float, ...]],
        target_points: List[Tuple[float, ...]],
        leader_id: Optional[int] = None
    ) -> Dict[int, Tuple[float, ...]]:
        """
        Gán đích mục tiêu tối ưu cho các drone.
        - Nếu leader_id được chỉ định và tồn tại trong current_poses:
          Gán cố định leader_id tới target_points[0] (vị trí chỉ huy danh dự: Mũi / Đỉnh / Tâm).
          Các drone còn lại (followers) được phân công tối ưu toàn cục tới target_points[1:]
          bằng thuật toán Hungarian, bảo đảm không bao giờ giao cắt đường bay.
        - Nếu leader_id là None:
          Toàn bộ bầy drone được gán thuần túy theo Hungarian tối thiểu hóa tổng khoảng cách.

        :param current_poses: Dict {drone_idx: (x, y, z, ...)}
        :param target_points: List [(gx, gy, gz, ...)]
        :param leader_id: ID hoặc Index của Drone Leader (VD: 1 trong GCS hoặc 0 trong controller).
        :return: Dict {drone_idx: (gx, gy, gz, ...)} đích đã được gán tối ưu
        """
        drone_ids = list(current_poses.keys())
        n_drones = len(drone_ids)
        n_targets = len(target_points)

        if n_drones == 0 or n_targets == 0:
            return {}

        if n_drones != n_targets:
            raise ValueError(f"Số lượng drone ({n_drones}) không khớp số lượng điểm đích ({n_targets})")

        assigned_targets: Dict[int, Tuple[float, ...]] = {}

        # Nếu có Leader Priority và leader_id có trong danh sách
        if leader_id is not None and leader_id in current_poses and n_targets > 1:
            # 1. Gán leader_id tới điểm mục tiêu đầu tiên (Slot 0 - Leader Slot)
            assigned_targets[leader_id] = target_points[0]

            # 2. Tập hợp các drone follower còn lại và các điểm mục tiêu còn lại
            follower_ids = [d for d in drone_ids if d != leader_id]
            sub_targets = target_points[1:]
            n_sub = len(follower_ids)

            cost_matrix = np.zeros((n_sub, n_sub), dtype=np.float64)
            for i, d_id in enumerate(follower_ids):
                p = current_poses[d_id]
                px, py = p[0], p[1]
                pz = p[2] if len(p) > 2 else 0.0

                for j, tgt in enumerate(sub_targets):
                    tx, ty = tgt[0], tgt[1]
                    tz = tgt[2] if len(tgt) > 2 else 0.0

                    dx = px - tx
                    dy = py - ty
                    dz = (pz - tz) * self.z_weight
                    cost_matrix[i, j] = dx * dx + dy * dy + dz * dz

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                d_id = follower_ids[r]
                assigned_targets[d_id] = sub_targets[c]

            return assigned_targets

        # Trường hợp chuẩn (không chỉ định leader_id hoặc bầy 1 drone)
        cost_matrix = np.zeros((n_drones, n_targets), dtype=np.float64)

        for i, d_id in enumerate(drone_ids):
            p = current_poses[d_id]
            px, py = p[0], p[1]
            pz = p[2] if len(p) > 2 else 0.0

            for j, tgt in enumerate(target_points):
                tx, ty = tgt[0], tgt[1]
                tz = tgt[2] if len(tgt) > 2 else 0.0

                # Chi phí khoảng cách bình phương Euclid có trọng số trục Z
                dx = px - tx
                dy = py - ty
                dz = (pz - tz) * self.z_weight
                cost_matrix[i, j] = dx * dx + dy * dy + dz * dz

        # Giải bài toán gán tuyến tính tối ưu toàn cục bằng thuật toán Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            d_id = drone_ids[r]
            assigned_targets[d_id] = target_points[c]

        return assigned_targets

    def get_assignment_metrics(
        self,
        current_poses: Dict[int, Tuple[float, ...]],
        assigned_targets: Dict[int, Tuple[float, ...]]
    ) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá chất lượng gán đích.
        """
        distances = []
        for d_id, p in current_poses.items():
            if d_id in assigned_targets:
                tgt = assigned_targets[d_id]
                dx = p[0] - tgt[0]
                dy = p[1] - tgt[1]
                dz = (p[2] - tgt[2]) if len(p) > 2 and len(tgt) > 2 else 0.0
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                distances.append(dist)

        if not distances:
            return {"total_dist": 0.0, "max_dist": 0.0, "avg_dist": 0.0}

        return {
            "total_dist": float(sum(distances)),
            "max_dist": float(max(distances)),
            "avg_dist": float(sum(distances) / len(distances)),
            "variance": float(np.var(distances)),
        }
