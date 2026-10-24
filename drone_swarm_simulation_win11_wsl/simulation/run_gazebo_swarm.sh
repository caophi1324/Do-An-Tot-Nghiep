#!/bin/bash
# =============================================================================
# KHỞI CHẠY GAZEBO HARMONIC 3D SWARM VISUALIZATION (WSL2 + Windows 11)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "🚀 KHỞI ĐỘNG GAZEBO HARMONIC 3D SWARM (WSL2 + WSLg)"
echo "======================================================"

COUNT="${1:-3}"

# 1. Cấu hình đồ họa ổn định cho WSLg (chống crash Segmentation fault của driver NVIDIA D3D12 libnvwgf2umx.so)
export LIBGL_ALWAYS_SOFTWARE=1
export vblank_mode=0

# 2. Cấu hình đường dẫn tài nguyên Model cho Gazebo
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$GZ_SIM_RESOURCE_PATH"

# 3. Dọn dẹp tiến trình cũ
pkill -9 -f "wsl_gazebo_swarm_bridge.py" 2>/dev/null || true

# 4. Sinh file Gazebo World SDF đúng số lượng drone
echo "🔧 Đang khởi tạo bản đồ Gazebo cho $COUNT Drones..."
python3 simulation/generate_gazebo_world.py "$COUNT"

# 5. Khởi động cầu nối MAVLink -> Gazebo Transport
echo "📡 Đang kích hoạt Gazebo Swarm Bridge ($COUNT Drones, 30 FPS)..."
python3 -u simulation/wsl_gazebo_swarm_bridge.py "$COUNT" &
BRIDGE_PID=$!
sleep 1

# 6. Mở giao diện 3D Gazebo Sim siêu mượt (Gazebo Harmonic engine)
echo "🖥️  Đang mở cửa sổ 3D Gazebo Sim trên màn hình Windows qua WSLg..."
gz sim -r simulation/gazebo_swarm_world.sdf

# 5. Dọn dẹp khi tắt cửa sổ Gazebo
echo ""
echo "🛑 Cửa sổ Gazebo đã đóng. Đang dọn dẹp các tiến trình ngầm..."
kill -9 $BRIDGE_PID 2>/dev/null || true
pkill -9 -f "wsl_gazebo_swarm_bridge.py" 2>/dev/null || true
echo "✅ Hoàn tất dọn dẹp!"
