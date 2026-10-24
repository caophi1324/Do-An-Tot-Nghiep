#!/bin/bash
# =============================================================================
# Dọn dẹp và tắt sạch sẽ 100% tất cả tiến trình 5 Drone SITL, MAVProxy, MAVROS
# =============================================================================

echo "🛑 Đang tắt toàn bộ tiến trình 5 Drone SITL, MAVProxy, MAVROS..."
killall -9 arducopter ArduCopter mavproxy.py sim_vehicle.py mavros_node xterm drone_3d_mesh_publisher.py 2>/dev/null || true
pkill -9 -f arducopter 2>/dev/null || true
pkill -9 -f mavproxy 2>/dev/null || true
pkill -9 -f sim_vehicle 2>/dev/null || true
pkill -9 -f mavros 2>/dev/null || true

echo "✅ ĐÃ TẮT SẠCH SẼ TẤT CẢ DRONE & MAVROS!"
