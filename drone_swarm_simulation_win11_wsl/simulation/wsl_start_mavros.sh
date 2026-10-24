#!/bin/bash
killall -9 mavros_node 2>/dev/null || true
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
fi

WIN_IP=$(grep nameserver /etc/resolv.conf | cut -d' ' -f2)
if [ -z "$WIN_IP" ]; then
    WIN_IP="172.28.240.1"
fi
echo "Connecting 5 MAVROS nodes to Windows Host: $WIN_IP..."

for i in 1 2 3 4 5; do
    port=$((5760 + i * 10 + 2))
    echo "  [Drone $i] MAVROS -> tcp://$WIN_IP:$port"
    ros2 run mavros mavros_node --ros-args -r __ns:=/drone$i --param fcu_url:=tcp://$WIN_IP:$port@ > /tmp/mavros_$i.log 2>&1 &
done

echo "All 5 MAVROS nodes launched. Running in background..."
wait
