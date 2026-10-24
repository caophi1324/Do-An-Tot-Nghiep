#!/usr/bin/env python3
"""
DRONE 3D MODEL PUBLISHER FOR RVIZ2 (REALISTIC FPV KAMIKAZE DRONE MESH)
Sử dụng mô hình 3D thực tế FPV Strike Kamikaze Drone (Collada .DAE / .OBJ)
- Nạp trực tiếp 3D Mesh nguyên bản kèm Texture độ nét cao
- Tích hợp vòng LED phát sáng nhận diện 5 màu (Đỏ, Xanh lá, Xanh dương, Vàng, Tím)
- 4 Đĩa cánh quạt quay 3D 30Hz sống động
- Nhãn tên Drone 3D (Text Billboard) hiển thị rõ tên từng drone
"""

import math
import os
import sys
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, Twist

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False

try:
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False


def rotate_vector_by_quaternion(v, q):
    """Xoay vector v=(x,y,z) theo quaternion q=(qx,qy,qz,qw)."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    vx, vy, vz = v

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)

    return rx, ry, rz


class Drone3DModelPublisher(Node):
    def __init__(self, num_drones=3):
        super().__init__('drone_3d_model_publisher')
        self.num_drones = num_drones
        self.prop_angle = 0.0
        self.running = True

        # Đường dẫn tuyệt đối tới file 3D Mesh (hỗ trợ cả Windows và WSL)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.normpath(os.path.join(current_dir, "..", "models", "kamikaze_drone"))
        dae_file = os.path.join(model_dir, "kamikaze_drone_ros.dae")
        
        # Format URI file:/// cho RViz2 Windows
        self.mesh_resource = "file:///" + dae_file.replace("\\", "/")
        self.get_logger().info(f"Loaded 3D Mesh Resource: {self.mesh_resource}")

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Toạ độ bãi đỗ mặt đất (Swarm Ground Offsets)
        # Mỗi drone có vị trí riêng biệt cách nhau 1.6m (span 6.4m), đồng bộ 100% với GCS Radar
        self.ground_offsets = {
            1: (0.0, 0.0, 0.0),    # Drone 1: Đỏ (X=0m, Y=0m)
            2: (0.0, 1.6, 0.0),    # Drone 2: Xanh lá (X=0m, Y=1.6m)
            3: (0.0, 3.2, 0.0),    # Drone 3: Xanh dương (X=0m, Y=3.2m)
            4: (0.0, 4.8, 0.0),    # Drone 4: Vàng (X=0m, Y=4.8m)
            5: (0.0, 6.4, 0.0),    # Drone 5: Tím (X=0m, Y=6.4m)
        }

        self.pubs = {}
        self.odom_pubs = {}
        self.raw_odom_pubs = {}
        self.subs = {}
        self.latest_poses = {}
        self.smooth_poses = {}
        self.target_poses = {}
        self.latest_meas_times = {}
        self.latest_twists = {}
        self.mav_connected = {}

        # 5 Màu sắc đặc trưng cho 5 drone
        self.drone_colors = [
            (1.00, 0.15, 0.15),  # Drone 1: Đỏ tươi (Bright Red)
            (0.15, 0.95, 0.25),  # Drone 2: Xanh lá (Bright Green)
            (0.20, 0.60, 1.00),  # Drone 3: Xanh dương (Bright Blue)
            (1.00, 0.90, 0.10),  # Drone 4: Vàng tươi (Bright Yellow)
            (0.85, 0.15, 0.85),  # Drone 5: Tím sáng (Bright Magenta)
        ]

        for i in range(1, num_drones + 1):
            topic_sub = f'/drone{i}/local_position/odom'
            topic_pub = f'/drone{i}/drone_model'
            topic_odom_pub = f'/drone{i}/odom_swarm'

            self.pubs[i] = self.create_publisher(MarkerArray, topic_pub, 10)
            self.odom_pubs[i] = self.create_publisher(Odometry, topic_odom_pub, 10)
            self.raw_odom_pubs[i] = self.create_publisher(Odometry, topic_sub, 10)
            self.mav_connected[i] = False

            self.subs[i] = self.create_subscription(
                Odometry,
                topic_sub,
                lambda msg, d_id=i: self.odom_callback(msg, d_id),
                qos_sub
            )
            # Khởi tạo toạ độ ban đầu theo bãi đỗ riêng biệt
            ox, oy, oz = self.ground_offsets.get(i, (0.0, float(i - 1) * 1.6, 0.0))
            default_p = Pose()
            default_p.position.x = ox
            default_p.position.y = oy
            default_p.position.z = oz
            default_p.orientation.w = 1.0
            self.latest_poses[i] = default_p

            sm_p = Pose()
            sm_p.position.x = ox
            sm_p.position.y = oy
            sm_p.position.z = oz
            sm_p.orientation.w = 1.0
            self.smooth_poses[i] = sm_p

            tgt_p = Pose()
            tgt_p.position.x = ox
            tgt_p.position.y = oy
            tgt_p.position.z = oz
            tgt_p.orientation.w = 1.0
            self.target_poses[i] = tgt_p

            self.latest_meas_times[i] = time.time()
            self.latest_twists[i] = Twist()

            # Khởi động luồng đọc MAVLink trực tiếp từ SITL Windows (SERIAL1 port)
            if HAS_PYMAVLINK:
                tcp_port = 5760 + i * 10 + 2  # D1:5772, D2:5782, D3:5792, D4:5802, D5:5812
                t = threading.Thread(target=self._mavlink_worker, args=(i, tcp_port), daemon=True)
                t.start()

            self.get_logger().info(f'Subscribed: {topic_sub} -> Publishing: {topic_pub} & {topic_odom_pub} (Offset: X={ox}m, Y={oy}m)')

        # Khởi tạo TF Broadcaster để RViz2 không bao giờ báo lỗi Frame [map] does not exist
        self.tf_broadcaster = TransformBroadcaster(self) if HAS_TF2 else None

        # Publisher cho đường nối Leader-Follower và vùng bảo vệ APF
        self.swarm_links_pub = self.create_publisher(MarkerArray, '/swarm/topology_links', 10)

        # Publisher cho Quả Cầu An Toàn 3D (Safety Spheres) cho bầy drone 5-inch
        self.safety_spheres_pub = self.create_publisher(MarkerArray, '/swarm/safety_spheres', 10)

        # Timer quay cánh quạt và nội suy mượt 50Hz (20ms)
        self.timer = self.create_timer(1.0 / 50.0, self.timer_callback)

    def _mavlink_worker(self, drone_id: int, port: int):
        """Đọc trực tiếp dữ liệu toạ độ và thái độ từ ArduPilot SITL qua TCP MAVLink (Không cần MAVROS)."""
        conn_str = f"tcp:127.0.0.1:{port}"
        cur_yaw_ros = 0.0
        cur_pitch_ros = 0.0
        cur_roll_ros = 0.0

        while self.running and rclpy.ok():
            mav = None
            try:
                mav = mavutil.mavlink_connection(conn_str, autoreconnect=False, retries=0)
                hb = mav.wait_heartbeat(timeout=2.0)
                if not hb:
                    mav.close()
                    time.sleep(1.0)
                    continue

                self.mav_connected[drone_id] = True
                self.get_logger().info(f"[Drone {drone_id}] Direct MAVLink connected ({conn_str})")

                # Cung cấp toạ độ EKF Origin chuẩn cho indoor drone
                try:
                    lat = int(-35.3632621 * 1e7)
                    lon = int((149.1652374 + drone_id * 0.0001) * 1e7)
                    alt = int(584 * 1000)
                    mav.mav.set_gps_global_origin_send(mav.target_system, lat, lon, alt)
                except Exception:
                    pass

                # Yêu cầu SITL stream toàn bộ dữ liệu 50Hz cho độ mịn tối đa
                mav.mav.request_data_stream_send(
                    mav.target_system,
                    mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    50,
                    1
                )

                ox, oy, oz = self.ground_offsets.get(drone_id, (0.0, float(drone_id - 1) * 1.6, 0.0))

                while self.running and rclpy.ok():
                    msg = mav.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE', 'HEARTBEAT'], blocking=True, timeout=1.0)
                    if msg is None:
                        continue

                    m_type = msg.get_type()
                    if m_type == 'ATTITUDE':
                        # Chuyển đổi thái độ ArduPilot NED sang ROS ENU
                        cur_yaw_ros = (math.pi / 2.0) - msg.yaw
                        cur_pitch_ros = -msg.pitch
                        cur_roll_ros = msg.roll

                        cy = math.cos(cur_yaw_ros * 0.5)
                        sy = math.sin(cur_yaw_ros * 0.5)
                        cp = math.cos(cur_pitch_ros * 0.5)
                        sp = math.sin(cur_pitch_ros * 0.5)
                        cr = math.cos(cur_roll_ros * 0.5)
                        sr = math.sin(cur_roll_ros * 0.5)

                        q = Quaternion()
                        q.w = cr * cp * cy + sr * sp * sy
                        q.x = sr * cp * cy - cr * sp * sy
                        q.y = cr * sp * cy + sr * cp * sy
                        q.z = cr * cp * sy - sr * sp * cy

                        if drone_id in self.target_poses:
                            self.target_poses[drone_id].orientation = q
                        if drone_id in self.smooth_poses:
                            self.smooth_poses[drone_id].orientation = q
                        if drone_id in self.latest_poses:
                            self.latest_poses[drone_id].orientation = q

                    elif m_type == 'LOCAL_POSITION_NED':
                        # Chuyển đổi toạ độ ArduPilot NED sang ROS ENU
                        ros_x = float(msg.y)   # East -> X_ros
                        ros_y = float(msg.x)   # North -> Y_ros
                        ros_z = float(-msg.z)  # Up -> Z_ros

                        target_p = Pose()
                        target_p.position.x = ros_x + ox
                        target_p.position.y = ros_y + oy
                        target_p.position.z = ros_z + oz
                        if drone_id in self.target_poses:
                            target_p.orientation = self.target_poses[drone_id].orientation
                        else:
                            target_p.orientation.w = 1.0

                        self.target_poses[drone_id] = target_p
                        self.latest_meas_times[drone_id] = time.time()

                        # Cập nhật vận tốc
                        tw = Twist()
                        tw.linear.x = float(msg.vy)
                        tw.linear.y = float(msg.vx)
                        tw.linear.z = float(-msg.vz)
                        self.latest_twists[drone_id] = tw

                        # Phát thêm vào /droneX/local_position/odom để tương thích các công cụ ROS 2 khác
                        if drone_id in self.raw_odom_pubs:
                            stamp = self.get_clock().now().to_msg()
                            local_odom = Odometry()
                            local_odom.header.stamp = stamp
                            local_odom.header.frame_id = 'map'
                            local_odom.child_frame_id = f'drone{drone_id}'
                            local_odom.pose.pose.position.x = ros_x
                            local_odom.pose.pose.position.y = ros_y
                            local_odom.pose.pose.position.z = ros_z
                            local_odom.pose.pose.orientation = target_p.orientation
                            local_odom.twist.twist = tw
                            self.raw_odom_pubs[drone_id].publish(local_odom)

            except Exception:
                time.sleep(1.0)
            finally:
                self.mav_connected[drone_id] = False
                if mav is not None:
                    try:
                        mav.close()
                    except Exception:
                        pass

    def destroy_node(self):
        self.running = False
        super().destroy_node()

    def odom_callback(self, msg: Odometry, drone_id: int):
        # Nếu đang nhận trực tiếp từ SITL MAVLink thì bỏ qua callback lặp
        if self.mav_connected.get(drone_id, False):
            return

        ox, oy, oz = self.ground_offsets.get(drone_id, (0.0, float(drone_id - 1) * 1.6, 0.0))
        world_pose = Pose()
        world_pose.position.x = msg.pose.pose.position.x + ox
        world_pose.position.y = msg.pose.pose.position.y + oy
        world_pose.position.z = msg.pose.pose.position.z + oz
        world_pose.orientation = msg.pose.pose.orientation
        self.target_poses[drone_id] = world_pose
        self.latest_meas_times[drone_id] = time.time()
        self.latest_twists[drone_id] = msg.twist.twist

    def timer_callback(self):
        self.prop_angle = (self.prop_angle + 0.8) % (2 * math.pi)
        stamp = self.get_clock().now().to_msg()
        now = time.time()

        # BỘ LỌC NỘI SUY MƯỢT DEAD RECKONING & LERP 50Hz
        for d_id in range(1, self.num_drones + 1):
            tgt_p = self.target_poses.get(d_id)
            if tgt_p is None:
                continue
            sm_p = self.smooth_poses.setdefault(d_id, tgt_p)
            tw = self.latest_twists.get(d_id, Twist())
            t_pkt = self.latest_meas_times.get(d_id, now)

            # 1. Dự đoán vi bước Dead Reckoning dựa trên vận tốc
            dt_lag = min(0.12, max(0.0, now - t_pkt))
            pred_x = tgt_p.position.x + tw.linear.x * dt_lag
            pred_y = tgt_p.position.y + tw.linear.y * dt_lag
            pred_z = tgt_p.position.z + tw.linear.z * dt_lag

            # 2. Bộ lọc thông thấp hàm mũ Exponential Moving Average (EMA)
            alpha = 0.35
            sm_p.position.x += alpha * (pred_x - sm_p.position.x)
            sm_p.position.y += alpha * (pred_y - sm_p.position.y)
            sm_p.position.z += alpha * (pred_z - sm_p.position.z)
            sm_p.orientation = tgt_p.orientation

            self.latest_poses[d_id] = sm_p

            # Phát Odometry mượt mà chuẩn thế giới cho RViz2
            if d_id in self.odom_pubs:
                swarm_odom = Odometry()
                swarm_odom.header.stamp = stamp
                swarm_odom.header.frame_id = 'map'
                swarm_odom.child_frame_id = f'drone{d_id}'
                swarm_odom.pose.pose = sm_p
                swarm_odom.twist.twist = tw
                self.odom_pubs[d_id].publish(swarm_odom)

        # Phát cây toạ độ TF2 (map -> drone1..5) cho RViz2 nhận diện Fixed Frame map mượt mà
        if self.tf_broadcaster:
            tf_msgs = []
            for d_id, pose in self.latest_poses.items():
                if pose is None:
                    continue
                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = 'map'
                t.child_frame_id = f'drone{d_id}'
                t.transform.translation.x = pose.position.x
                t.transform.translation.y = pose.position.y
                t.transform.translation.z = pose.position.z
                t.transform.rotation = pose.orientation
                tf_msgs.append(t)
            if tf_msgs:
                self.tf_broadcaster.sendTransform(tf_msgs)

        for d_id, pose in self.latest_poses.items():
            if pose is None:
                continue

            markers = self.create_quadcopter_markers(d_id, pose)
            self.pubs[d_id].publish(markers)

        # Phát đường nối topo bầy đàn và vùng bảo vệ APF
        self.publish_swarm_topology(stamp)

        # Phát quả cầu an toàn 3D (Safety Spheres) cho drone 5-inch
        self.publish_safety_spheres(stamp)

    def publish_swarm_topology(self, stamp):
        """Phát đường nối ảo Leader-Follower và vùng bảo vệ APF trực quan trên RViz2"""
        marker_array = MarkerArray()

        # 1. Đường nối Leader-Follower (LINE_LIST màu Cyber Cyan)
        lines_marker = Marker()
        lines_marker.header.frame_id = 'map'
        lines_marker.header.stamp = stamp
        lines_marker.ns = 'swarm_topology'
        lines_marker.id = 100
        lines_marker.type = Marker.LINE_LIST
        lines_marker.action = Marker.ADD
        lines_marker.scale.x = 0.035
        lines_marker.color.r = 0.1
        lines_marker.color.g = 0.9
        lines_marker.color.b = 0.8
        lines_marker.color.a = 0.75

        leader_pose = self.latest_poses.get(1)
        if leader_pose:
            p_lead = Point(x=leader_pose.position.x, y=leader_pose.position.y, z=leader_pose.position.z)
            for d_id in range(2, self.num_drones + 1):
                f_pose = self.latest_poses.get(d_id)
                if f_pose:
                    p_fol = Point(x=f_pose.position.x, y=f_pose.position.y, z=f_pose.position.z)
                    lines_marker.points.append(p_lead)
                    lines_marker.points.append(p_fol)
        marker_array.markers.append(lines_marker)

        # 2. Cảnh báo khoảng cách gần / Lực đẩy APF (LINE_LIST màu Đỏ/Cam)
        repulsion_marker = Marker()
        repulsion_marker.header.frame_id = 'map'
        repulsion_marker.header.stamp = stamp
        repulsion_marker.ns = 'apf_repulsion'
        repulsion_marker.id = 101
        repulsion_marker.type = Marker.LINE_LIST
        repulsion_marker.action = Marker.ADD
        repulsion_marker.scale.x = 0.06
        repulsion_marker.color.r = 1.0
        repulsion_marker.color.g = 0.2
        repulsion_marker.color.b = 0.1
        repulsion_marker.color.a = 0.95

        for i in range(1, self.num_drones + 1):
            for j in range(i + 1, self.num_drones + 1):
                p_i = self.latest_poses.get(i)
                p_j = self.latest_poses.get(j)
                if p_i and p_j:
                    dx = p_i.position.x - p_j.position.x
                    dy = p_i.position.y - p_j.position.y
                    dz = p_i.position.z - p_j.position.z
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist < 1.20:  # Bán kính ảnh hưởng APF d_influence chuẩn 1.20m cho hội trường 20x30m
                        pt1 = Point(x=p_i.position.x, y=p_i.position.y, z=p_i.position.z)
                        pt2 = Point(x=p_j.position.x, y=p_j.position.y, z=p_j.position.z)
                        repulsion_marker.points.append(pt1)
                        repulsion_marker.points.append(pt2)
        self.swarm_links_pub.publish(marker_array)

    def publish_safety_spheres(self, stamp):
        """Phát quả cầu an toàn 3D (Safety Sphere) cho từng drone 5-inch để kiểm chứng trực quan va chạm."""
        marker_array = MarkerArray()

        # Drone 5-inch geometry constants
        # Bán kính bao cánh quạt vật lý: ~0.18 - 0.20m (đường kính 0.40m)
        # Bán kính an toàn khí động học ORCA (Safety bubble): 0.40m (đường kính 0.80m)
        SPHERE_RADIUS = 0.40
        SPHERE_DIAMETER = SPHERE_RADIUS * 2.0  # 0.80m
        PHYSICAL_CRITICAL_DIST = 0.40          # 2 * 0.20m = 0.40m: Cánh quạt 5-inch chém vào nhau
        ORCA_BREACH_DIST = 0.80                # 2 * 0.40m = 0.80m: Hai quả cầu an toàn cắt nhau
        PROXIMITY_WARN_DIST = 1.00             # Tiếp cận cận biên an toàn

        # 1. Tính toán cự ly nhỏ nhất giữa từng drone với các drone còn lại trong bầy
        min_distances = {}
        closest_pairs = {}
        for i in range(1, self.num_drones + 1):
            p_i = self.latest_poses.get(i)
            if not p_i:
                continue
            min_d = float('inf')
            closest_j = None
            for j in range(1, self.num_drones + 1):
                if i == j:
                    continue
                p_j = self.latest_poses.get(j)
                if not p_j:
                    continue
                dx = p_i.position.x - p_j.position.x
                dy = p_i.position.y - p_j.position.y
                dz = p_i.position.z - p_j.position.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < min_d:
                    min_d = dist
                    closest_j = j
            min_distances[i] = min_d
            closest_pairs[i] = closest_j

        # 2. Tạo Marker SPHERE cho từng drone 5-inch (Khóa TF frame drone{d_id} mượt mà)
        for d_id in range(1, self.num_drones + 1):
            pose = self.latest_poses.get(d_id)
            sphere = Marker()
            sphere.header.frame_id = f'drone{d_id}'
            sphere.header.stamp = stamp
            sphere.ns = 'safety_spheres'
            sphere.id = d_id
            sphere.type = Marker.SPHERE

            if not pose:
                sphere.action = Marker.DELETE
                marker_array.markers.append(sphere)
                continue

            sphere.action = Marker.ADD
            sphere.frame_locked = True
            sphere.pose.position.x = 0.0
            sphere.pose.position.y = 0.0
            sphere.pose.position.z = 0.0
            sphere.pose.orientation.w = 1.0

            sphere.scale.x = SPHERE_DIAMETER
            sphere.scale.y = SPHERE_DIAMETER
            sphere.scale.z = SPHERE_DIAMETER

            r, g, b = self.drone_colors[(d_id - 1) % len(self.drone_colors)]
            min_d = min_distances.get(d_id, float('inf'))

            # Phân cấp cảnh báo màu 4 mức thông minh theo cự ly thực tế
            if min_d < PHYSICAL_CRITICAL_DIST:
                # Mức 4: Cực kỳ nguy hiểm - Cánh quạt 5-inch chém vào nhau
                # Chớp sáng đỏ rực 6Hz
                flash = (int(time.time() * 6) % 2 == 0)
                sphere.color.r = 1.0
                sphere.color.g = 0.0
                sphere.color.b = 0.0
                sphere.color.a = 0.80 if flash else 0.40
            elif min_d < ORCA_BREACH_DIST:
                # Mức 3: Vi phạm vùng an toàn ORCA (2 quả cầu cắt lồng vào nhau)
                sphere.color.r = 1.0
                sphere.color.g = 0.15
                sphere.color.b = 0.15
                sphere.color.a = 0.55  # Đỏ rực
            elif min_d < PROXIMITY_WARN_DIST:
                # Mức 2: Cận biên an toàn - Hai quả cầu sắp chạm nhau
                sphere.color.r = 1.0
                sphere.color.g = 0.55
                sphere.color.b = 0.0
                sphere.color.a = 0.38  # Vàng Cam cảnh báo
            else:
                # Mức 1: An toàn danh định - Bong bóng phát sáng màu riêng của drone
                sphere.color.r = r
                sphere.color.g = g
                sphere.color.b = b
                sphere.color.a = 0.22  # Trong suốt nhẹ, nhìn thấu rõ mô hình 3D và cánh quạt bên trong

            marker_array.markers.append(sphere)

        # 3. Tạo đường laser đỏ và nhãn hiển thị cự ly vi phạm nếu khoảng cách < 0.80m
        breach_line = Marker()
        breach_line.header.frame_id = 'map'
        breach_line.header.stamp = stamp
        breach_line.ns = 'safety_breach'
        breach_line.id = 100
        breach_line.type = Marker.LINE_LIST
        breach_line.scale.x = 0.05
        breach_line.color.r = 1.0
        breach_line.color.g = 0.05
        breach_line.color.b = 0.05
        breach_line.color.a = 0.90

        breach_label = Marker()
        breach_label.header.frame_id = 'map'
        breach_label.header.stamp = stamp
        breach_label.ns = 'safety_breach'
        breach_label.id = 101
        breach_label.type = Marker.TEXT_VIEW_FACING
        breach_label.scale.z = 0.14
        breach_label.color.r = 1.0
        breach_label.color.g = 0.1
        breach_label.color.b = 0.1
        breach_label.color.a = 0.95

        min_overall_breach = float('inf')
        breach_midpoint = None

        for i in range(1, self.num_drones + 1):
            for j in range(i + 1, self.num_drones + 1):
                p_i = self.latest_poses.get(i)
                p_j = self.latest_poses.get(j)
                if p_i and p_j:
                    dx = p_i.position.x - p_j.position.x
                    dy = p_i.position.y - p_j.position.y
                    dz = p_i.position.z - p_j.position.z
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist < ORCA_BREACH_DIST:
                        pt1 = Point(x=p_i.position.x, y=p_i.position.y, z=p_i.position.z)
                        pt2 = Point(x=p_j.position.x, y=p_j.position.y, z=p_j.position.z)
                        breach_line.points.append(pt1)
                        breach_line.points.append(pt2)
                        if dist < min_overall_breach:
                            min_overall_breach = dist
                            breach_midpoint = (
                                (p_i.position.x + p_j.position.x) * 0.5,
                                (p_i.position.y + p_j.position.y) * 0.5,
                                (p_i.position.z + p_j.position.z) * 0.5 + 0.35
                            )

        if len(breach_line.points) > 0 and breach_midpoint:
            breach_line.action = Marker.ADD
            breach_label.action = Marker.ADD
            breach_label.pose.position.x = breach_midpoint[0]
            breach_label.pose.position.y = breach_midpoint[1]
            breach_label.pose.position.z = breach_midpoint[2]
            if min_overall_breach < PHYSICAL_CRITICAL_DIST:
                breach_label.text = f"! PROP CRASH ! {min_overall_breach:.2f}m"
            else:
                breach_label.text = f"! ORCA BREACH ! {min_overall_breach:.2f}m"
        else:
            breach_line.action = Marker.DELETE
            breach_label.action = Marker.DELETE

        marker_array.markers.append(breach_line)
        marker_array.markers.append(breach_label)

        self.safety_spheres_pub.publish(marker_array)

    def create_quadcopter_markers(self, drone_id: int, pose) -> MarkerArray:
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        r, g, b = self.drone_colors[(drone_id - 1) % len(self.drone_colors)]

        q = pose.orientation
        px, py, pz = pose.position.x, pose.position.y, pose.position.z

        # --- 1. MÔ HÌNH 3D THỰC TẾ (FPV Strike Kamikaze Drone Mesh) ---
        # Khóa cứng vào frame drone{drone_id} với frame_locked = True
        # GPU / TF2 đảm nhận chuyển động mượt mà, triệt tiêu 100% việc nạp lại file Collada 6.14MB
        mesh_m = Marker()
        mesh_m.header.frame_id = f'drone{drone_id}'
        mesh_m.header.stamp = stamp
        mesh_m.ns = f'drone{drone_id}'
        mesh_m.id = 0
        mesh_m.type = Marker.MESH_RESOURCE
        mesh_m.mesh_resource = self.mesh_resource
        mesh_m.mesh_use_embedded_materials = True
        mesh_m.action = Marker.ADD
        mesh_m.frame_locked = True
        mesh_m.pose.position.x = 0.0
        mesh_m.pose.position.y = 0.0
        mesh_m.pose.position.z = 0.0
        mesh_m.pose.orientation.w = 1.0
        # Scale phóng to nhẹ 1.3x để nhìn rõ các chi tiết khung sườn, đầu đạn và camera
        mesh_m.scale.x = 1.3
        mesh_m.scale.y = 1.3
        mesh_m.scale.z = 1.3
        marker_array.markers.append(mesh_m)

        # --- 2. VÒNG LED ĐÈN HIỆU MÀU NHẬN DIỆN (LED Beacon Halo) ---
        led = Marker()
        led.header.frame_id = f'drone{drone_id}'
        led.header.stamp = stamp
        led.ns = f'drone{drone_id}'
        led.id = 1
        led.type = Marker.CYLINDER
        led.action = Marker.ADD
        led.frame_locked = True
        led.pose.position.x = 0.0
        led.pose.position.y = 0.0
        led.pose.position.z = 0.12
        led.pose.orientation.w = 1.0
        led.scale.x = 0.22
        led.scale.y = 0.22
        led.scale.z = 0.06
        led.color.r = r
        led.color.g = g
        led.color.b = b
        led.color.a = 0.90
        marker_array.markers.append(led)

        # --- 3. 4 ĐĨA CÁNH QUẠT QUAY 3D (Spinning Propeller Disks) ---
        # Toạ độ 4 motor trên mô hình FPV Kamikaze Drone (scale 1.3x)
        motor_pos = [
            (0.24, -0.28, 0.06),   # 0: Trước Phải
            (0.24,  0.28, 0.06),   # 1: Trước Trái
            (-0.24, -0.28, 0.06),  # 2: Sau Phải
            (-0.24,  0.28, 0.06),  # 3: Sau Trái
        ]

        for m_idx, (lx, ly, lz) in enumerate(motor_pos):
            prop = Marker()
            prop.header.frame_id = f'drone{drone_id}'
            prop.header.stamp = stamp
            prop.ns = f'drone{drone_id}'
            prop.id = 10 + m_idx
            prop.type = Marker.CYLINDER
            prop.action = Marker.ADD
            prop.frame_locked = True
            prop.pose.position.x = lx
            prop.pose.position.y = ly
            prop.pose.position.z = lz + 0.02
            prop.pose.orientation.w = 1.0
            prop.scale.x = 0.26
            prop.scale.y = 0.26
            prop.scale.z = 0.015
            # Màu đĩa quay phản quang theo màu drone
            prop.color.r = min(1.0, r + 0.2)
            prop.color.g = min(1.0, g + 0.2)
            prop.color.b = min(1.0, b + 0.2)
            prop.color.a = 0.55
            marker_array.markers.append(prop)

        # --- 4. NHÃN TÊN & CAO ĐỘ 3D DRONE (Tactical Aerospace HUD Tag) ---
        # Thiết kế thanh thoát: 1 nhãn duy nhất ở đỉnh drone, không đè lấp mô hình khi bay sát nhau
        label = Marker()
        label.header.frame_id = 'map'
        label.header.stamp = stamp
        label.ns = f'drone{drone_id}'
        label.id = 20
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = px
        label.pose.position.y = py
        label.pose.position.z = pz + 0.30  # Đặt cố định trên đỉnh drone trong hệ map
        label.scale.z = 0.10              # Kích thước nhỏ gọn 10cm thanh thoát
        label.color.r = r
        label.color.g = g
        label.color.b = b
        label.color.a = 0.95
        label.text = f"D{drone_id} • {pz:.1f}m"
        marker_array.markers.append(label)

        return marker_array


def main():
    rclpy.init()
    num_drones = 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        num_drones = int(sys.argv[1])
    node = Drone3DModelPublisher(num_drones=num_drones)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
