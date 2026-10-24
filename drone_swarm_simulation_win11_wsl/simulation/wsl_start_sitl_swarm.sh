#!/bin/bash
# =============================================================================
# WSL: Launch 4-Drone Swarm SITL via official sim_vehicle.py (Headless Mode)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
HOME_LOC="-35.3632621,149.1652374,584,353"

# 1. Activate python virtualenv with MAVProxy
if [ -f "$HOME/venv-ardupilot/bin/activate" ]; then
    source "$HOME/venv-ardupilot/bin/activate"
fi

# 2. Cleanup old processes & stale EEPROM files
echo "🔄 Dọn dẹp process & file EEPROM cũ..."
killall -9 arducopter mavproxy.py sim_vehicle.py mavros_node xterm 2>/dev/null || true
rm -f "$ARDUPILOT_DIR/eeprom*.bin" "$ARDUPILOT_DIR/ArduCopter/eeprom*.bin" "$HOME"/eeprom*.bin 2>/dev/null || true
sleep 2

# 3. Disable extra GUI xterm windows to save 100% CPU
export AUTOTEST_TERM=none
unset DISPLAY
export DISPLAY=""

# 4. Launch multi-drone swarm (5 vehicles, spaced 5m apart, headless with clean EEPROM)
HARDWARE_PARM="$SCRIPT_DIR/config/sitl_hardware_gy912_mtf02.parm"
PARM_ARG=""
if [ -f "$HARDWARE_PARM" ]; then
    echo "📦 Nạp thông số phần cứng thực tế (GY-912 10DOF + MTF-02 ToF 2.5m/Optical Flow)..."
    PARM_ARG="--add-param-file=$HARDWARE_PARM"
fi

COUNT="${1:-3}"
echo "🚀 Đang khởi động $COUNT Drones SITL sạch sẽ (sim_vehicle.py -w)..."
nohup python3 "$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py" \
    -w \
    -v ArduCopter -f "+" --no-rebuild \
    --count="$COUNT" \
    --auto-sysid \
    --auto-offset-line=90,5 \
    -l "$HOME_LOC" \
    --speedup=1 \
    $PARM_ARG \
    --mavproxy-args="--daemon" \
    > "/tmp/sitl_swarm.log" 2>&1 &


SITL_PID=$!
disown $SITL_PID 2>/dev/null || true
echo "   SITL Background PID: $SITL_PID"
echo "⏳ Đang đợi 5 drone khởi tạo và hiệu chuẩn Gyro/Baro (15 giây)..."
sleep 15

echo ""
echo "======================================================"
echo "✅ 5 Drones SITL đã sẵn sàng (CPU nhẹ mượt)!"
echo "   Drone 1 -> MAVROS: tcp://127.0.0.1:5762 | Python: tcp:127.0.0.1:5763"
echo "   Drone 2 -> MAVROS: tcp://127.0.0.1:5772 | Python: tcp:127.0.0.1:5773"
echo "   Drone 3 -> MAVROS: tcp://127.0.0.1:5782 | Python: tcp:127.0.0.1:5783"
echo "   Drone 4 -> MAVROS: tcp://127.0.0.1:5792 | Python: tcp:127.0.0.1:5793"
echo "   Drone 5 -> MAVROS: tcp://127.0.0.1:5802 | Python: tcp:127.0.0.1:5803"
echo "======================================================"