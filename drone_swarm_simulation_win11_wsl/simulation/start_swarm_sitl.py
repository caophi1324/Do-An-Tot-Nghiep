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

# Mặc định 3 Drone (có thể truyền số lượng drone qua dòng lệnh: python start_swarm_sitl.py 5)
num_drones = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
processes = []

hardware_parm = Path(__file__).resolve().parent.parent / "config" / "sitl_hardware_gy912_mtf02.parm"
if hardware_parm.exists():
    defaults_arg = f"{default_params},{hardware_parm}"
else:
    defaults_arg = str(default_params)

model_json = Path(__file__).resolve().parent.parent / "models" / "micro_drone_240g.json"
if model_json.exists():
    model_arg = f"+:{model_json.as_posix()}"
else:
    model_arg = "+"

print("========================================")
print(f"  Tạo SWARM {num_drones} Drones Ảo (Hybrid Mode)")
print("  Vật lý: Micro Drone AUW 240g (1404 Motor, 3-inch Prop, 5.8A Hover)")
print("  Tích hợp: GY-912 (10DOF) + MTF-02 (ToF 2.5m)")
print("  SERIAL0 TCP -> GCS/debug")
print("  SERIAL1 UDP -> MAVROS (ROS2)")
print("  SERIAL2 TCP -> Python Script")
print("========================================")
print("Bấm Ctrl+C để tắt toàn bộ drones.\n")

try:
    for i in range(1, num_drones + 1):
        tcp_serial0  = 5760 + i*10       # D1:5770, D2:5780, D3:5790, D4:5800, D5:5810  (GCS / Debug)
        tcp_serial1  = tcp_serial0 + 2   # D1:5772, D2:5782, D3:5792, D4:5802, D5:5812  (MAVROS TCP)
        tcp_serial2  = tcp_serial0 + 3   # D1:5773, D2:5783, D3:5793, D4:5803, D5:5813  (Python Mission TCP)

        print(f"[*] Đang khởi động Drone {i} (SYSID {i})...")
        print(f"    - SERIAL0 GCS    : tcp:127.0.0.1:{tcp_serial0}")
        print(f"    - SERIAL1 MAVROS : tcp:127.0.0.1:{tcp_serial1}")
        print(f"    - SERIAL2 Python : tcp:127.0.0.1:{tcp_serial2}")

        home_lon = 149.1652374 + (i * 0.0001)
        home_str = f"-35.3632621,{home_lon},584,353"

        # Mỗi instance phải chạy trong thư mục riêng để không bị đụng độ file eeprom.bin và logs
        i_dir = sitl_dir / str(i)
        i_dir.mkdir(parents=True, exist_ok=True)

        # Xóa eeprom.bin cũ để ArduPilot 100% nạp lại thông số mới đã tune từ defaults .parm
        eeprom_file = i_dir / "eeprom.bin"
        if eeprom_file.exists():
            try:
                eeprom_file.unlink()
            except Exception:
                pass

        cmd = [
            str(arducopter_exe),
            "-w",                       # Wipe EEPROM nạp mới sạch sẽ toàn bộ thông số
            "-I", str(i),
            "--sysid", str(i),
            "--model", model_arg,
            "--speedup", "1",
            "--defaults", defaults_arg,
            "--home", home_str,
            "--serial0", "tcp:0",
            "--serial1", "tcp:2",
            "--serial2", "tcp:3",
        ]

        p = subprocess.Popen(cmd, cwd=str(i_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(1.0)  # Đợi 1.0s để ArduCopter khởi tạo xong trước khi bật con tiếp theo

    print(f"\n✅ Đã bật thành công {num_drones} Drones (Hybrid Mode)!")

    print("\n" + "="*50)
    print("[WSL] Lệnh chạy MAVROS bằng TCP (Copy và dán vào WSL):")
    print("export WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')")
    for i in range(1, num_drones + 1):
        tcp_port = 5760 + i*10 + 2
        print(f"ros2 run mavros mavros_node --ros-args -r __ns:=/drone{i} --param fcu_url:=tcp://$WIN_IP:{tcp_port}@ &")

    print("\n[Windows] Chạy kịch bản bay biểu diễn 30 năm thành lập trường:")
    print("python missions\\show_30years_smooth.py")
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
