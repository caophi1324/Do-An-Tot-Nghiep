#!/bin/bash
# =============================================================================
# Launch 5-Drone MAVROS Bridge + 3D Mesh Visualizer + RViz2 GUI in WSL2
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "🚀 KHỞI ĐỘNG RVIZ2 3D SWARM & MAVROS BRIDGE (WSL2)"
echo "======================================================"

# 1. Nạp ROS 2 Jazzy
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "❌ Không tìm thấy ROS 2 Jazzy tại /opt/ros/jazzy/setup.bash"
    exit 1
fi

# 2. Lấy IP Host Windows để MAVROS kết nối tới ArduCopter SITL
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
if [ -z "$WIN_IP" ]; then
    WIN_IP="127.0.0.1"
fi
echo "🔗 Địa chỉ Host Windows: $WIN_IP"

# 3. Dọn dẹp các tiến trình cũ
echo "🧹 Đang dọn dẹp các tiến trình MAVROS/RViz2 cũ..."
killall -9 rviz2 mavros_node drone_3d_mesh_publisher.py 2>/dev/null || true
sleep 1

# 4. Khởi động 5 MAVROS node cho 5 drone
echo "📡 Đang kết nối 5 MAVROS node tới ArduCopter SITL..."
for i in {1..5}; do
    port=$((5760 + i * 10 + 2))
    echo "   [Drone $i] MAVROS -> tcp://$WIN_IP:$port"
    ros2 run mavros mavros_node --ros-args -r __ns:=/drone$i --param fcu_url:=tcp://$WIN_IP:$port@ > /dev/null 2>&1 &
done
sleep 2

# 5. Khởi động 3D Mesh Publisher (Cánh quạt quay 30Hz, đèn LED, cây toạ độ TF map)
echo "🛸 Đang kích hoạt 3D Model & TF Broadcaster..."
python3 visualization/drone_3d_mesh_publisher.py 5 > /dev/null 2>&1 &
sleep 1

# 6. Mở RViz2 trực quan hoá
echo "🖥️  Đang mở giao diện RViz2 GUI trên màn hình Windows..."
echo "💡 (Khi bạn đóng cửa sổ RViz2, script sẽ tự động dọn dẹp toàn bộ tiến trình ngầm)."
rviz2 -d config/swarm_view.rviz

# 7. Dọn dẹp khi tắt RViz2
echo ""
echo "🛑 RViz2 đã đóng. Đang tắt toàn bộ tiến trình ngầm..."
killall -9 mavros_node drone_3d_mesh_publisher.py 2>/dev/null || true
echo "✅ Hoàn tất dọn dẹp!"
