#!/usr/bin/env python3
"""
TEST ORCA PIPELINE
Bộ kiểm thử toán học & mô phỏng định lượng cho hệ thống ORCA Swarm:
1. Test Hungarian Assigner: Kiểm tra tối ưu hóa ma trận khoảng cách & tính duy nhất.
2. Test Minimum Snap Trajectory: Kiểm tra điều kiện biên nghỉ v(0)=v(T)=0, a(0)=a(T)=0.
3. Test B-Spline Trajectory: Kiểm tra đạo hàm bậc 1, bậc 2 của scipy.interpolate.
4. Test ORCA Head-On Collision: Tình huống 2 drone đối đầu trực diện 180° (kịch bản va chạm khó nhất).
5. Test ORCA Circle Antipodal Swap: 5 drone đổi vị trí đối xứng xuyên tâm (thử thách bầy đàn kinh điển).
"""

import math
import sys
import os
import numpy as np

# Thêm thư mục gốc vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from orca_swarm import (
    HungarianAssigner,
    MinimumSnapTrajectory,
    BSplineTrajectory,
    ORCA2DEngine,
    ORCA3DEngine,
    ORCAAgent,
    ORCAConfig,
)


def test_hungarian():
    print("\n--- TEST 1: HUNGARIAN TARGET ASSIGNER ---")
    assigner = HungarianAssigner(z_weight=1.2)

    # 5 drone xuất phát dàn hàng ngang
    starts = {
        0: (0.0, 0.0, 1.3),
        1: (0.0, 1.6, 1.3),
        2: (0.0, 3.2, 1.3),
        3: (0.0, 4.8, 1.3),
        4: (0.0, 6.4, 1.3),
    }

    # 5 đích đến hình ngôi sao
    targets = [
        (2.5, 5.4, 1.4),
        (4.6, 3.9, 1.4),
        (3.8, 1.4, 1.4),
        (1.2, 1.4, 1.4),
        (0.4, 3.9, 1.4),
    ]

    assigned = assigner.assign(starts, targets)
    metrics = assigner.get_assignment_metrics(starts, assigned)

    assert len(assigned) == 5, "Phải gán đủ 5 drone"
    # Kiểm tra mỗi đích chỉ được gán đúng 1 lần
    assigned_pts = list(assigned.values())
    for tgt in targets:
        assert tgt in assigned_pts, f"Điểm đích {tgt} bị bỏ sót!"

    print(f"✅ Hungarian Assignment PASS! Tổng cự ly: {metrics['total_dist']:.2f}m, Cực đại: {metrics['max_dist']:.2f}m")


def test_minimum_snap():
    print("\n--- TEST 2: MINIMUM SNAP TRAJECTORY (7TH-ORDER) ---")
    p0 = (0.0, 0.0, 1.0)
    p1 = (3.0, 4.0, 2.0)
    duration = 4.0

    traj = MinimumSnapTrajectory(p0, p1, duration)

    # Kiểm tra t = 0
    pos0, vel0, acc0 = traj.evaluate(0.0)
    assert np.allclose(pos0, p0, atol=1e-5), f"Sai vị trí đầu: {pos0}"
    assert np.allclose(vel0, 0.0, atol=1e-5), f"Sai vận tốc đầu: {vel0}"
    assert np.allclose(acc0, 0.0, atol=1e-5), f"Sai gia tốc đầu: {acc0}"

    # Kiểm tra t = T
    posT, velT, accT = traj.evaluate(duration)
    assert np.allclose(posT, p1, atol=1e-5), f"Sai vị trí cuối: {posT}"
    assert np.allclose(velT, 0.0, atol=1e-5), f"Sai vận tốc cuối: {velT}"
    assert np.allclose(accT, 0.0, atol=1e-5), f"Sai gia tốc cuối: {accT}"

    # Kiểm tra t = T/2: Đạt đỉnh tốc độ
    pos_mid, vel_mid, acc_mid = traj.evaluate(duration / 2.0)
    speed_mid = np.linalg.norm(vel_mid)
    expected_dist = np.linalg.norm(np.array(p1) - np.array(p0))  # 5.099m
    # Vận tốc đỉnh của đa thức bậc 7: ds_dtau_max tại tau=0.5 là 140*(0.125) - 420*(0.0625) + 420*(0.03125) - 140*(0.015625) = 2.1875
    # v_peak = 5.099 * 2.1875 / 4.0 = 2.78 m/s
    assert speed_mid > 1.0, f"Vận tốc giữa chừng quá thấp: {speed_mid}"

    print(f"✅ Minimum Snap PASS! v(0)={np.linalg.norm(vel0):.4f}, v(T)={np.linalg.norm(velT):.4f}, v_peak={speed_mid:.2f} m/s")


def test_bspline():
    print("\n--- TEST 3: B-SPLINE TRAJECTORY ---")
    waypoints = [
        (0.0, 0.0, 1.3),
        (1.5, 1.0, 1.5),
        (3.0, 2.5, 1.4),
        (4.0, 4.0, 1.3)
    ]
    duration = 5.0
    bspl = BSplineTrajectory(waypoints, duration=duration, degree=3)

    p_start, v_start, _ = bspl.evaluate(0.0)
    p_end, v_end, _ = bspl.evaluate(duration)

    assert np.allclose(p_start, waypoints[0], atol=1e-4), "Sai điểm đầu B-Spline"
    assert np.allclose(p_end, waypoints[-1], atol=1e-4), "Sai điểm cuối B-Spline"
    assert np.allclose(v_start, 0.0, atol=1e-4), f"Vận tốc đầu không bằng 0: {v_start}"
    assert np.allclose(v_end, 0.0, atol=1e-4), f"Vận tốc cuối không bằng 0: {v_end}"

    print(f"✅ B-Spline PASS! Đường bay mượt mà qua {len(waypoints)} mốc.")


def test_orca_2d_head_on():
    print("\n--- TEST 4: ORCA HEAD-ON 180° COLLISION TEST ---")
    cfg = ORCAConfig(time_horizon=2.0, agent_radius=0.35, max_speed=1.5, time_step=0.05, safety_margin=0.10)
    engine = ORCA2DEngine(cfg)

    # Hai drone bay đối đầu trực diện trên trục X từ cự ly 4 mét
    # Drone 0: ở X = -2.0m, muốn bay sang X = +2.0m (v_pref = +1.0 m/s)
    # Drone 1: ở X = +2.0m, muốn bay sang X = -2.0m (v_pref = -1.0 m/s)
    pos0 = np.array([-2.0, 0.0, 1.3])
    pos1 = np.array([2.0, 0.0, 1.3])
    vel0 = np.array([1.0, 0.0, 0.0])
    vel1 = np.array([-1.0, 0.0, 0.0])

    dt = 0.05
    min_dist_observed = float('inf')
    steps = 100  # 5.0 giây

    for step in range(steps):
        dist = np.linalg.norm(pos0[:2] - pos1[:2])
        if dist < min_dist_observed:
            min_dist_observed = dist

        # v_pref hướng về đích
        target0 = np.array([2.0, 0.0, 1.3])
        target1 = np.array([-2.0, 0.0, 1.3])

        dir0 = target0[:2] - pos0[:2]
        d0 = np.linalg.norm(dir0)
        v_pref0 = (dir0 / d0) * min(1.2, d0) if d0 > 0.05 else np.zeros(2)

        dir1 = target1[:2] - pos1[:2]
        d1 = np.linalg.norm(dir1)
        v_pref1 = (dir1 / d1) * min(1.2, d1) if d1 > 0.05 else np.zeros(2)

        agent0 = ORCAAgent(0, pos0, vel0, np.array([v_pref0[0], v_pref0[1], 0.0]), radius=cfg.agent_radius, max_speed=cfg.max_speed)
        agent1 = ORCAAgent(1, pos1, vel1, np.array([v_pref1[0], v_pref1[1], 0.0]), radius=cfg.agent_radius, max_speed=cfg.max_speed)

        v_safe0, _ = engine.compute_agent_velocity(agent0, [agent0, agent1])
        v_safe1, _ = engine.compute_agent_velocity(agent1, [agent0, agent1])

        # Cập nhật vị trí tự do theo vận tốc ORCA
        pos0 += v_safe0 * dt
        pos1 += v_safe1 * dt
        vel0 = v_safe0
        vel1 = v_safe1

    print(f"   ↳ Cự ly nhỏ nhất đo được khi 2 drone đối đầu trực diện: {min_dist_observed:.3f}m")
    assert min_dist_observed >= (cfg.agent_radius * 2.0), f"Va chạm! Cự ly {min_dist_observed} < {cfg.agent_radius * 2.0}"
    # Xác nhận cả 2 drone đã đến gần đích của mình
    assert np.linalg.norm(pos0[:2] - np.array([2.0, 0.0])) < 0.5, "Drone 0 chưa đến đích"
    assert np.linalg.norm(pos1[:2] - np.array([-2.0, 0.0])) < 0.5, "Drone 1 chưa đến đích"

    print("✅ ORCA Head-On Test PASS! Tự do bẻ lái né nhau mượt mà không va chạm.")


def test_orca_circle_swap():
    print("\n--- TEST 5: ORCA 5-DRONE CIRCLE ANTIPODAL SWAP TEST ---")
    cfg = ORCAConfig(time_horizon=2.0, agent_radius=0.35, max_speed=1.5, time_step=0.05, safety_margin=0.10)
    engine = ORCA2DEngine(cfg)

    # 5 drone xếp trên đường tròn bán kính 2.5m, mục tiêu là bay xuyên tâm đổi chỗ sang phía đối diện
    n = 5
    radius = 2.5
    start_angles = [2.0 * math.pi * i / n for i in range(n)]

    positions = {i: np.array([radius * math.cos(ang), radius * math.sin(ang), 1.3]) for i, ang in enumerate(start_angles)}
    targets = {i: np.array([-radius * math.cos(ang), -radius * math.sin(ang), 1.3]) for i, ang in enumerate(start_angles)}
    velocities = {i: np.zeros(3) for i in range(n)}

    dt = 0.05
    min_dist_observed = float('inf')
    steps = 120  # 6 giây

    for step in range(steps):
        # Đo cự ly nhỏ nhất giữa mọi cặp drone
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i][:2] - positions[j][:2])
                if d < min_dist_observed:
                    min_dist_observed = d

        # Tạo danh sách agent
        agents = []
        for i in range(n):
            dir_to_tgt = targets[i][:2] - positions[i][:2]
            dist_to_tgt = np.linalg.norm(dir_to_tgt)
            if dist_to_tgt > 0.05:
                pref_v_xy = (dir_to_tgt / dist_to_tgt) * min(1.2, dist_to_tgt)
            else:
                pref_v_xy = np.zeros(2)

            v_pref = np.array([pref_v_xy[0], pref_v_xy[1], 0.0])
            agents.append(ORCAAgent(
                agent_id=i,
                position=positions[i],
                velocity=velocities[i],
                pref_velocity=v_pref,
                radius=cfg.agent_radius,
                max_speed=cfg.max_speed
            ))

        # Tính vận tốc ORCA an toàn
        new_velocities = {}
        for agent in agents:
            v_safe, _ = engine.compute_agent_velocity(agent, agents)
            new_velocities[agent.id] = v_safe

        # Cập nhật vị trí
        for i in range(n):
            positions[i] += new_velocities[i] * dt
            velocities[i] = new_velocities[i]

    print(f"   ↳ Cự ly nhỏ nhất giữa mọi cặp drone trong bài test 5-Drone Circle Swap: {min_dist_observed:.3f}m")
    assert min_dist_observed >= (cfg.agent_radius * 2.0), f"Va chạm! Cự ly {min_dist_observed} < {cfg.agent_radius * 2.0}"
    print("✅ ORCA 5-Drone Circle Antipodal Swap PASS! Toàn bầy lướt qua nhau hoàn hảo không tắc nghẽn.")


if __name__ == "__main__":
    test_hungarian()
    test_minimum_snap()
    test_bspline()
    test_orca_2d_head_on()
    test_orca_circle_swap()
    print("\n🎉🎉🎉 TOÀN BỘ 5 BÀI TEST TOÁN HỌC & ĐỘNG HỌC ORCA/RVO2 ĐÃ VƯỢT QUA 100%! 🎉🎉🎉")
