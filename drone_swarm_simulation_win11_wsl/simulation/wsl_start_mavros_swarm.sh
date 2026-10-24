#!/bin/bash
# =============================================================================
# WSL: Launch 4 MAVROS nodes for RViz2 Visualization
# Connects to SERIAL1 TCP ports: 5762, 5772, 5782, 5792
# =============================================================================

set -e

source /opt/ros/jazzy/setup.bash

echo "🔄 Dọn dẹp MAVROS cũ..."
killall -9 mavros_node 2>/dev/null || true
sleep 1

NUM_DRONES=5
echo "🚀 Đang khởi động $NUM_DRONES MAVROS nodes..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🎨 Đang khởi động Drone 3D Mesh Publisher ($NUM_DRONES drones)..."
nohup python3 "$SCRIPT_DIR/drone_3d_mesh_publisher.py" $NUM_DRONES > /tmp/mesh_pub.log 2>&1 &
MESH_PID=$!
disown $MESH_PID 2>/dev/null || true
sleep 1

for i in $(seq 0 $((NUM_DRONES - 1))); do
    SYSID=$((i + 1))
    TCP_PORT=$((5762 + i * 10))   # 5762, 5772, 5782, 5792, 5802

    echo "  [Drone $SYSID] MAVROS -> tcp://127.0.0.1:$TCP_PORT (namespace: /drone$SYSID)"
    nohup ros2 run mavros mavros_node --ros-args \
        -r __ns:=/drone$SYSID \
        --param fcu_url:=tcp://127.0.0.1:$TCP_PORT \
        --param tgt_system:=$SYSID \
        --param system_id:=$((200 + SYSID)) \
        --param conn.timeout:=10.0 \
        --param conn.heartbeat_rate:=1.0 \
        --param plugin_denylist:="['param', 'waypoint', 'mission', 'ftp', 'distance_sensor', 'log_transfer', 'terrain', 'adsb', 'companion_process', 'debug_value', 'esc_status', 'esc_telemetry', 'gps_input', 'gps_rtk', 'gpsstatus', 'landing_target', 'mag_calibration', 'mount_control', 'obstacle', 'obstacle_distance_3d', 'play_tune', 'rangefinder', 'trajectory', 'vfr_hud', 'vibration', 'vision_pose', 'vision_speed', 'mocap', 'fake_gps', 'px4flow', 'open_drone_id', 'gimbal_control', 'geofence', 'rallypoint', 'tunnel', 'tdr_radio', 'wheel_odometry', 'wind_estimation', 'camera', 'cellular_status']" \
        > "/tmp/mavros_drone${SYSID}.log" 2>&1 &
    MAV_PID=$!
    disown $MAV_PID 2>/dev/null || true
    sleep 2.5
done

echo "⏳ Đang đợi 5 MAVROS nodes nạp plugin và kết nối ổn định (18 giây)..."
sleep 18

# Kích hoạt stream rate 10Hz (msg 32: LOCAL_POSITION_NED, msg 30: ATTITUDE) cho từng drone đồng thời qua Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/activate_mavros_streams.py" $NUM_DRONES || true




echo ""
echo "======================================================"
echo "✅ Tất cả 5 MAVROS nodes đã chạy và streaming!"
echo "   Kiểm tra topics: ros2 topic list | grep local_position"
echo "   Mở RViz2:        rviz2"
echo "======================================================"