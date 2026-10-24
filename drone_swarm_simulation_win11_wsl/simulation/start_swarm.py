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

num_drones = 4
processes = []

print("========================================")
print(f"  Tạo SWARM {num_drones} Drones Ảo")
print("========================================")
print("Bấm Ctrl+C để tắt toàn bộ drones.\n")

try:
    for i in range(1, num_drones + 1):
        port = 5760 + i*10
        # Mặc dù dùng Instance i (1,2,3,4) nhưng SYSID sẽ ép về 1,2,3,4 để dễ theo dõi
        print(f"[*] Đang khởi động Drone {i} (SYSID {i}, Port {port})...")
        # Khởi tạo mỗi drone cách nhau 10m về phía đông để không bị đụng nhau khi RTL
        home_lon = 149.1652374 + (i * 0.0001)
        home_str = f"-35.3632621,{home_lon},584,353"

        cmd = [
            str(arducopter_exe),
            "-I", str(i),
            "--sysid", str(i),
            "--model", "+",
            "--speedup", "1",
            "--defaults", str(default_params),
            "--home", home_str
        ]
        
        # Chạy ẩn ArduCopter.exe (không hiện cửa sổ cmd gây rối mắt)
        p = subprocess.Popen(cmd, cwd=str(sitl_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(1)  # Đợi 1 giây giữa các lần bật để tránh nghẽn CPU
        
    print("\n✅ Đã bật thành công 4 Drones!")
    print("- Drone 1 chờ kết nối ở: tcp:127.0.0.1:5770")
    print("- Drone 2 chờ kết nối ở: tcp:127.0.0.1:5780")
    print("- Drone 3 chờ kết nối ở: tcp:127.0.0.1:5790")
    print("- Drone 4 chờ kết nối ở: tcp:127.0.0.1:5800")
    print("\n>>> Bây giờ bạn mở terminal mới và chạy: scripts\\demo_swarm_highspeed.py")
    print("\n(Script này vẫn đang chạy ngầm để giữ 4 drone hoạt động. Bấm Ctrl+C để tắt).")
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nĐang tắt toàn bộ drone ảo...")
    for p in processes:
        p.terminate()
    print("Đã tắt xong!")
