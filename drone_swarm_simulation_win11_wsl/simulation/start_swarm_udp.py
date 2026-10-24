import subprocess
import time
import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Thư mục chứa SITL của Mission Planner
sitl_dir = Path(os.environ['USERPROFILE']) / "Documents" / "Mission Planner" / "sitl"
arducopter_exe = sitl_dir / "ArduCopter.exe"
default_params = sitl_dir / "default_params" / "copter.parm"

if not arducopter_exe.exists():
    print(f"Error: Không tìm thấy file giả lập ở {arducopter_exe}")
    print("Bạn đã từng chạy tab Simulation trong Mission Planner ít nhất 1 lần chưa?")
    sys.exit(1)

num_drones = 5
processes = []

hardware_parm = Path(__file__).resolve().parent.parent / "config" / "sitl_hardware_gy912_mtf02.parm"
if hardware_parm.exists():
    defaults_arg = f"{default_params},{hardware_parm}"
else:
    defaults_arg = str(default_params)

print("========================================")
print(f"  Tạo SWARM {num_drones} Drones Ảo (Hybrid Mode)")
print("  Tích hợp: GY-912 (10DOF) + MTF-02 (ToF 2.5m)")
print("  SERIAL0 TCP -> GCS/debug")
print("  SERIAL1 UDP -> MAVROS (ROS2)")
print("  SERIAL2 TCP -> Python Script")
print("========================================")
print("Bấm Ctrl+C để tắt toàn bộ drones.\n")

try:
    for i in range(1, num_drones + 1):
        tcp_serial0  = 5760 + i*10       # D1:5770, D2:5780, D3:5790, D4:5800, D5:5810  (GCS/debug)
        tcp_serial2  = tcp_serial0 + 2   # D1:5772, D2:5782, D3:5792, D4:5802, D5:5812  (Python)
        udp_mavros   = 14560 + (i-1)*10  # D1:14560, D2:14570, D3:14580, D4:14590, D5:14600 (MAVROS)

        print(f"[*] Đang khởi động Drone {i} (SYSID {i})...")
        print(f"    - TCP Python : tcp:127.0.0.1:{tcp_serial2}")
        print(f"    - UDP MAVROS : udp:0.0.0.0:{udp_mavros}")

        home_lon = 149.1652374 + (i * 0.0001)
        home_str = f"-35.3632621,{home_lon},584,353"

        cmd = [
            str(arducopter_exe),
            "-I", str(i),
            "--sysid", str(i),
            "--model", "+",
            "--speedup", "1",
            "--defaults", defaults_arg,
            "--home", home_str,
            # SERIAL1 mở UDP server cho MAVROS kết nối vào
            "--serial1", f"udpserver:0.0.0.0:{udp_mavros}",
        ]

        p = subprocess.Popen(cmd, cwd=str(sitl_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(1.5)  # Đợi 1.5s để ArduCopter khởi tạo xong trước khi bật con tiếp theo

    print(f"\n✅ Đã bật thành công {num_drones} Drones (Hybrid Mode)!")

    print("\n" + "="*50)
    print("[WSL] Lệnh chạy MAVROS bằng UDP (Copy và dán vào WSL):")
    print("export WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')")
    for i in range(1, num_drones + 1):
        udp_mavros = 14560 + (i-1)*10
        print(f"ros2 run mavros mavros_node --ros-args -r __ns:=/drone{i} --param fcu_url:=udp://$WIN_IP:{udp_mavros}@ &")

    print("\n[Windows] Chạy kịch bản bay biểu diễn:")
    print(".venv\\Scripts\\python.exe missions\\demo_swarm_circle_stair.py")

    print("\n[WSL] Bật RViz2 để xem trực quan:")
    print("rviz2")
    print("  -> Global Options -> Fixed Frame: map")
    print("  -> Add -> By topic -> /droneX/local_position/odom -> Reliability: Best Effort")
    print("="*50)
    print("\n(Script đang chạy ngầm. Bấm Ctrl+C để tắt toàn bộ).")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nĐang tắt toàn bộ drone ảo...")
    for p in processes:
        p.terminate()
    print("Đã tắt xong!")
