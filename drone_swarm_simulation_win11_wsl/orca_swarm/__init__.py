"""
ORCA SWARM PACKAGE
Hệ thống điều khiển bầy đàn đa Drone thế hệ mới:
- Hungarian Algorithm: Gán mục tiêu tối ưu toàn cục (scipy.optimize.linear_sum_assignment)
- B-Spline & Minimum Snap: Tạo đường bay và vector vận tốc mong muốn mượt mà cấp C²/C³
- ORCA / RVO2 (Optimal Reciprocal Collision Avoidance): Lõi né tránh va chạm tương hỗ không gian vận tốc
- ORCA Swarm Controller: Điều phối bầy đàn khép vòng Telemetry MAVLink thời gian thực (10Hz)
"""

from .hungarian_assigner import HungarianAssigner
from .trajectory_generator import MinimumSnapTrajectory, BSplineTrajectory, TrajectoryGenerator
from .orca_rvo2_engine import ORCA2DEngine, ORCA3DEngine, ORCAAgent, ORCAConfig
from .orca_controller import ORCASwarmController

__all__ = [
    "HungarianAssigner",
    "MinimumSnapTrajectory",
    "BSplineTrajectory",
    "TrajectoryGenerator",
    "ORCA2DEngine",
    "ORCA3DEngine",
    "ORCAAgent",
    "ORCAConfig",
    "ORCASwarmController",
]
