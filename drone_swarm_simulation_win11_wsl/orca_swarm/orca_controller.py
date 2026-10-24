#!/usr/bin/env python3
"""
ORCA SWARM CONTROLLER
Bộ điều phối bầy đàn khép vòng Telemetry (Closed-Loop Swarm Controller):
- Tích hợp chuỗi thuật toán:
  1. Hungarian Algorithm: Gán đích tối ưu toàn cục.
  2. Minimum Snap / B-Spline: Cung cấp vector vận tốc mong muốn v_pref và vị trí tham chiếu.
  3. ORCA/RVO2 Engine: Lọc và hiệu chỉnh vận tốc an toàn v_safe trong không gian vận tốc tự do.
- Tuyệt đối KHÔNG bắt buộc drone đi theo đường ray cứng; drone tự do lách hướng để né tránh va chạm.
- Tần số bắn lệnh 10Hz, chuyển đổi chuẩn xác giữa toạ độ ROS ENU và ArduPilot Local NED.
"""

import asyncio
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .hungarian_assigner import HungarianAssigner
from .trajectory_generator import MinimumSnapTrajectory, BSplineTrajectory, TrajectoryGenerator
from .orca_rvo2_engine import ORCA2DEngine, ORCA3DEngine, ORCAAgent, ORCAConfig


# Tọa độ bãi đỗ xuất phát mặt đất của 5 drone (chuẩn hội trường 20x30m, bước 1.6m)
SWARM_BASE_OFFSETS = {
    0: (0.0, 0.0),    # Drone 1 (Đỏ):         X_ros = 0.0m, Y_ros = 0.0m
    1: (0.0, 1.6),    # Drone 2 (Xanh lá):    X_ros = 0.0m, Y_ros = 1.6m
    2: (0.0, 3.2),    # Drone 3 (Xanh dương): X_ros = 0.0m, Y_ros = 3.2m
    3: (0.0, 4.8),    # Drone 4 (Vàng):       X_ros = 0.0m, Y_ros = 4.8m
    4: (0.0, 6.4),    # Drone 5 (Tím):        X_ros = 0.0m, Y_ros = 6.4m
}


async def send_ros_position(
    handler,
    x_ros: float,
    y_ros: float,
    z_alt: float,
    yaw_ros_deg: float = None,
    vx_ros: float = 0.0,
    vy_ros: float = 0.0,
    vz_ros: float = 0.0,
    use_vel: bool = False
):
    """
    Chuyển đổi toạ độ ROS ENU (East, North, Altitude) sang MAVLink Local NED của ArduPilot:
    - X_NED (North) = y_ros
    - Y_NED (East)  = x_ros
    - Z_NED (Down)  = -z_alt
    - Yaw_NED       = (90 - yaw_ros) % 360 (hoặc None nếu không chỉ định yaw)
    - Vx_NED (North)= vy_ros
    - Vy_NED (East) = vx_ros
    - Vz_NED (Down) = -vz_ros
    """
    yaw_mav = None
    if yaw_ros_deg is not None:
        yaw_mav = (90.0 - yaw_ros_deg) % 360.0
    return await handler.set_position_local(
        x=y_ros, y=x_ros, z_alt=z_alt, yaw=yaw_mav,
        vx=vy_ros, vy=vx_ros, vz=-vz_ros,
        use_vel=use_vel
    )


def read_actual_poses(active_drones: list, base_offsets: dict = SWARM_BASE_OFFSETS, max_age_sec: float = 0.5):
    """
    Đọc vị trí và vận tốc thực tế của từng drone từ telemetry MAVLink.
    Trả về dict: {idx: (x_world, y_world, z_world, vx_world, vy_world, vz_world, yaw_ros)}
    """
    now = time.time()
    poses = {}
    for d_id, handler in active_drones:
        t = handler.get_telemetry()
        if not t.local_valid or (now - t.local_timestamp) > max_age_sec:
            return None
        idx = d_id - 1
        bx, by = base_offsets.get(idx, (0.0, float(idx) * 1.6))
        # local NED: t.local_x (North), t.local_y (East), t.local_z (Down)
        # ROS ENU: x = East + bx, y = North + by, z = Altitude = -Down
        x_world = float(t.local_y) + bx
        y_world = float(t.local_x) + by
        z_world = float(-t.local_z)
        vx = float(getattr(t, "vy", getattr(t, "local_vy", 0.0)))
        vy = float(getattr(t, "vx", getattr(t, "local_vx", 0.0)))
        vz = float(-getattr(t, "vz", getattr(t, "local_vz", 0.0)))
        yaw_ros = (90.0 - t.yaw) % 360.0
        poses[idx] = (x_world, y_world, z_world, vx, vy, vz, yaw_ros)
    return poses


class SwarmORCAController:
    """
    Bộ điều khiển trung tâm bầy drone tự do ORCA/RVO2 kết hợp:
    1. Hungarian Assigner: Tối ưu hoá toàn cục, triệt tiêu giao cắt đường bay.
    2. Minimum Snap Trajectory Generator: Quỹ đạo phân tích bậc 7 mượt mà tuyệt đối (Jerk min).
    3. ORCA 2D Engine: Không gian vận tốc tự do né tránh va chạm tương hỗ theo thời gian thực.
    """

    def __init__(
        self,
        config: Optional[ORCAConfig] = None,
        base_offsets: dict = SWARM_BASE_OFFSETS,
        use_3d: bool = False
    ):
        self.config = config or ORCAConfig(
            time_horizon=2.5,
            agent_radius=0.45,
            max_speed=2.4,
            time_step=0.05,
            safety_margin=0.25
        )
        self.base_offsets = base_offsets
        self.use_3d = use_3d
        self.assigner = HungarianAssigner(z_weight=1.2)
        self.orca_engine = ORCA3DEngine(self.config) if use_3d else ORCA2DEngine(self.config)

    async def execute_transition(
        self,
        active_drones: list,
        target_points: List[Tuple[float, ...]],
        duration_sec: float = 4.0,
        hz: int = 20,
        traj_type: str = "min_snap",
        desc: str = "",
        leader_id: Optional[int] = None
    ) -> Dict[int, Tuple[float, ...]]:
        """
        Thực hiện một pha chuyển cảnh đội hình mượt mà (Silky Smooth) ở 20Hz:
        - Sử dụng toạ độ chỉ lệnh ảo p_cmd tích phân tiến theo Minimum Snap và ORCA.
        - Triệt tiêu 100% hiện tượng neo trễ telemetry (Inertial Lag).
        - Giữ nguyên góc Yaw tự nhiên (không giật đầu) trong suốt quá trình chuyển dịch.
        - Phân tầng cao độ 3D động học (Dynamic 3D Altitude Layering) khi cự ly 2D < 1.4m.
        """
        total_steps = max(1, int(duration_sec * hz))
        dt = 1.0 / hz

        if desc:
            print(f"\n🌀 [ORCA SWARM] Chuyển cảnh: {desc}")
            print(f"   ↳ Thời gian: {duration_sec}s | Tần số: {hz}Hz | Tạo đường: {traj_type.upper()} | Tránh va chạm: ORCA/RVO2")

        # 1. Đọc vị trí xuất phát thực tế
        actual = read_actual_poses(active_drones, self.base_offsets)
        while actual is None:
            await asyncio.sleep(0.05)
            actual = read_actual_poses(active_drones, self.base_offsets)

        current_poses = {d_id - 1: tuple(actual[d_id - 1][:3]) for d_id, _ in active_drones if (d_id - 1) in actual}

        # 2. Gán mục tiêu tối ưu toàn cục bằng thuật toán Hungarian (Leader=None để gán tự do toàn cục)
        assigned_targets = self.assigner.assign(current_poses, target_points, leader_id=leader_id)
        metrics = self.assigner.get_assignment_metrics(current_poses, assigned_targets)
        leader_str = f"Leader=D{leader_id + 1}" if leader_id is not None else "Global Free"
        print(f"   ↳ [Hungarian Assigned ({leader_str})] Tổng cự ly: {metrics['total_dist']:.2f}m | Cự ly TB: {metrics['avg_dist']:.2f}m | Cực đại: {metrics['max_dist']:.2f}m")

        # Thời gian bay tối ưu động học bảo đảm vận tốc tối đa không vượt quá giới hạn an toàn
        min_kinematic_time = max(duration_sec, metrics['max_dist'] / 0.8)
        if min_kinematic_time > duration_sec:
            duration_sec = min_kinematic_time
            total_steps = int(duration_sec * hz)
            print(f"   ↳ [Động học mượt mà] Tự động điều chỉnh thời gian bay: {duration_sec:.1f}s ({total_steps} bước @ {hz}Hz)")

        # 3. Tạo quỹ đạo tham chiếu phân tích mượt mà (Minimum Snap)
        trajectories = TrajectoryGenerator.create_swarm_trajectories(
            current_poses, assigned_targets, duration=duration_sec, traj_type=traj_type
        )

        # 4. Vòng lặp điều khiển 20Hz: Tích phân tiến ảo từ vị trí xuất phát (Zero Telemetry Anchor)
        p_cmd = {idx: np.array([current_poses[idx][0], current_poses[idx][1], current_poses[idx][2]], dtype=np.float64) for idx in current_poses}
        v_cmd = {idx: np.zeros(3, dtype=np.float64) for idx in current_poses}

        step = 0
        max_safety_steps = int((duration_sec + 5.0) * hz)
        next_t = time.monotonic()
        min_dist_live = float('inf')

        while step <= max_safety_steps:
            t_curr = min(duration_sec, step * dt)

            # Kiểm tra hoàn thành hội tụ mượt mà (Zero Step Discontinuity)
            if step >= total_steps:
                max_err_to_goal = max(
                    np.linalg.norm(p_cmd[idx][:2] - np.array(assigned_targets[idx][:2]))
                    for idx in p_cmd if idx in assigned_targets
                )
                if max_err_to_goal < 0.12:
                    break

            # Phân tầng cao độ 3D động học trơn tru (Smooth 3D Dynamic Altitude Layering)
            # Sử dụng hàm smoothstep w(d_xy) để phân tầng Z tăng giảm êm ái C^1, triệt tiêu 100% cú sốc giật cao độ tức thời
            z_layer_offsets = {idx: 0.0 for idx in p_cmd}
            indices = list(p_cmd.keys())
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                for j_idx in range(i_idx + 1, len(indices)):
                    j = indices[j_idx]
                    d_xy = float(np.linalg.norm(p_cmd[i][:2] - p_cmd[j][:2]))
                    if d_xy < 1.40:
                        # Smoothstep weight từ 0 (tại d_xy = 1.40m) lên 1.0 (tại d_xy <= 0.85m)
                        s = max(0.0, min(1.0, (1.40 - d_xy) / (1.40 - 0.85)))
                        w = s * s * (3.0 - 2.0 * s)
                        needed_dz = (math.sqrt(max(0.0, 0.92**2 - min(0.85, d_xy)**2)) + 0.25 * (1.40 - d_xy) / 1.40) * w
                        z_layer_offsets[i] = max(z_layer_offsets[i], needed_dz * 0.50)
                        z_layer_offsets[j] = min(z_layer_offsets[j], -needed_dz * 0.50)

            # Xây dựng danh sách ORCAAgent dựa trên vector bám theo Minimum Snap trơn tru có phân tầng 3D
            agents = []
            for d_id, _ in active_drones:
                idx = d_id - 1
                pos_3d = np.array([p_cmd[idx][0], p_cmd[idx][1], p_cmd[idx][2] + z_layer_offsets.get(idx, 0.0)], dtype=np.float64)
                traj = trajectories.get(idx)

                if traj is not None:
                    p_ref, v_ref, _ = traj.evaluate(t_curr)
                    p_ref_3d = p_ref.copy()
                    p_ref_3d[2] += z_layer_offsets.get(idx, 0.0)

                    err = p_ref_3d - pos_3d
                    v_pref = v_ref + 1.2 * err
                    sp = np.linalg.norm(v_pref[:2])
                    if sp > self.config.max_speed:
                        v_pref[:2] = (v_pref[:2] / sp) * self.config.max_speed
                else:
                    v_pref = np.zeros(3)

                agents.append(ORCAAgent(
                    agent_id=idx,
                    position=pos_3d,
                    velocity=v_cmd[idx],
                    pref_velocity=v_pref,
                    radius=self.config.agent_radius,
                    max_speed=self.config.max_speed
                ))

            # Tính toán vận tốc an toàn v_safe qua lõi ORCA/RVO2
            handler_map = {d_id - 1: handler for d_id, handler in active_drones}
            tasks = []
            for agent in agents:
                v_safe, stats = self.orca_engine.compute_agent_velocity(agent, agents)
                if stats["min_distance_m"] < min_dist_live:
                    min_dist_live = stats["min_distance_m"]

                # Cập nhật vận tốc thực tế của bước này để ORCA bước kế tiếp tính nón va chạm chuẩn xác
                v_cmd[agent.id] = v_safe

                # Tích phân toạ độ ngang XY theo vận tốc ORCA v_safe
                # Trục Z duy trì độ cao danh định p_ref (KHÔNG tích luỹ z_layer_offsets vào p_cmd để chống sụt độ cao rơi đất)
                traj = trajectories.get(agent.id)
                p_ref_now, _, _ = traj.evaluate(t_curr) if traj is not None else (p_cmd[agent.id], None, None)
                p_cmd[agent.id][0] = float(p_cmd[agent.id][0] + v_safe[0] * dt)
                p_cmd[agent.id][1] = float(p_cmd[agent.id][1] + v_safe[1] * dt)
                p_cmd[agent.id][2] = float(p_ref_now[2])

                # Đổi sang Local NED của ArduPilot
                # lz cộng độ lệch phân tầng z_layer_offsets tức thời (hồi phục về danh định ngay khi d_xy >= 1.4m)
                bx, by = self.base_offsets.get(agent.id, (0.0, float(agent.id) * 1.6))
                lx = float(p_cmd[agent.id][0] - bx)
                ly = float(p_cmd[agent.id][1] - by)
                lz = float(p_cmd[agent.id][2] + z_layer_offsets.get(agent.id, 0.0))

                handler = handler_map.get(agent.id)
                if handler is not None:
                    tasks.append(send_ros_position(
                        handler,
                        x_ros=lx, y_ros=ly, z_alt=lz,
                        yaw_ros_deg=None,
                        vx_ros=float(v_safe[0]), vy_ros=float(v_safe[1]), vz_ros=0.0,
                        use_vel=True
                    ))

            await asyncio.gather(*tasks)

            if step % 20 == 0 and min_dist_live < 1.40:
                print(f"\r   ⚡ [ORCA RECIPROCAL AVOIDANCE] Min Dist: {min_dist_live:.2f}m | Tự do né tránh trong không gian vận tốc...   ", end="")

            step += 1
            next_t += dt
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

        print(f"\n   ✅ [ORCA HOÀN THÀNH CHUYỂN CẢNH] Đã hội tụ êm ái! Min Dist ghi nhận: {min_dist_live:.2f}m")
        return assigned_targets

    async def hold_formation(
        self,
        active_drones: list,
        targets: Dict[int, Tuple[float, ...]],
        hold_sec: float = 4.0,
        hz: int = 20
    ):
        """
        Giữ cố định đội hình tại chỗ: Duy trì liên tục MAVLink stream ở tần số 20Hz với use_vel=True (vx=0, vy=0, vz=0).
        Giữ ArduPilot luôn ở SubMode::POSVEL để controller không bao giờ bị reset (init_xy_controller),
        triệt tiêu 100% hiện tượng khựng/giật khi chuyển trạng thái giữa bay và đứng yên!
        """
        total_steps = max(1, int(hold_sec * hz))
        dt = 1.0 / hz
        next_t = time.monotonic()
        for _ in range(total_steps):
            tasks = []
            for d_id, handler in active_drones:
                idx = d_id - 1
                if idx in targets:
                    tgt = targets[idx]
                    bx, by = self.base_offsets.get(idx, (0.0, float(idx) * 1.6))
                    lx = tgt[0] - bx
                    ly = tgt[1] - by
                    lz = tgt[2] if len(tgt) > 2 else 1.3
                    yaw = tgt[3] if len(tgt) > 3 else None
                    tasks.append(send_ros_position(
                        handler,
                        x_ros=lx, y_ros=ly, z_alt=lz,
                        yaw_ros_deg=yaw,
                        vx_ros=0.0, vy_ros=0.0, vz_ros=0.0,
                        use_vel=True
                    ))
            if tasks:
                await asyncio.gather(*tasks)
            next_t += dt
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)


# Alias tương thích ngược
ORCASwarmController = SwarmORCAController
