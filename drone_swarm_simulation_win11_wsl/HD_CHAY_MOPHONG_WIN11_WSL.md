# HƯỚNG DẪN CHẠY MÔ PHỎNG & KỊCH BẢN BẦY DRONE (SWARM GCS)
### Dành cho Windows 11 & WSL2 (Ubuntu)

Tài liệu này hướng dẫn người nhận cài đặt và chạy mô phỏng bầy 5 Drone ArduPilot cùng các kịch bản bay tự hành (ORCA / RVO2, Hungarian, Minimum Snap, APF) và trạm điều khiển mặt đất (GCS).

---

## 📁 CẤU TRÚC GÓI MÔ PHỎNG
- `app_gcs/`: Ứng dụng GCS Desktop (PyQt6) tích hợp Radar chiến thuật, điều khiển đội hình và khởi chạy kịch bản.
- `orca_swarm/`: Engine thuật toán ORCA/RVO2 + Hungarian gán đích + Quỹ đạo Minimum Snap bậc 7.
- `missions/`: Các kịch bản bay tự hành (Show 30 năm, APF Leader-Follower, v.v.).
- `core/`: Thư viện lõi giao tiếp MAVLink thời gian thực và động cơ APF Swarm.
- `simulation/`: Script tự động khởi tạo cụm 5 Drone ảo ArduPilot SITL trên Windows & WSL.
- `visualization/`: Node ROS 2 kết nối MAVROS và hiển thị mô hình 3D trên RViz2.
- `models/`: Thông số vật lý Drone micro 240g và file 3D Mesh (.dae/.obj).
- `config/`: File thông số phần cứng SITL (.parm) và cấu hình RViz2 (.rviz).

---

## 🚀 CÁCH 1: CHẠY TRỰC TIẾP TRÊN WINDOWS 11 (ĐƠN GIẢN NHẤT)
*Dành cho trường hợp chỉ cần mở GCS Dashboard 2D Radar, xem dữ liệu Telemetry và chạy kịch bản bay.*

### Bước 1: Cài đặt Python & Thư viện
1. Cài đặt **Python 3.10, 3.11 hoặc 3.12** trên Windows (Nhớ tick chọn *"Add python.exe to PATH"* khi cài).
2. Mở Terminal (PowerShell hoặc CMD) tại thư mục giải nén:
   ```bash
   pip install -r requirements.txt
   ```

### Bước 2: Cài đặt ArduCopter SITL
- Cài phần mềm **Mission Planner** từ trang chủ ArduPilot: https://ardupilot.org/planner/docs/mission-planner-installation.html
- Mở Mission Planner lên 1 lần, vào tab **Simulation** và chọn **Multirotor** để Mission Planner tự tải `ArduCopter.exe` về máy (thư mục mặc định: `C:\Users\<Tên_User>\Documents\Mission Planner\sitl\`).
- Sau khi tải xong thì tắt Mission Planner.

### Bước 3: Khởi chạy 5 Drone ảo SITL
Mở Terminal tại thư mục dự án và chạy:
```bash
python simulation/start_swarm_sitl.py
```
*(Cụm 5 Drone ảo sẽ được bật ngầm và lắng nghe tại các cổng TCP 5770-5810).*

### Bước 4: Mở GCS và Chạy Kịch bản
Nhấp đúp chuột vào file:
- **`RUN_SWARM_GCS.bat`** (hoặc `RUN_ORCA_GCS.bat`)

Trên giao diện GCS:
1. Quan sát góc trên bên phải: khi hiển thị **"5/5 DRONES CONNECTED"** màu xanh là đã kết nối xong.
2. Bấm nút **"⚔️ ARM TOÀN BỘ"** -> **"🛫 CẤT CÁNH ĐỒNG BỘ"** (1.3m).
3. Trong mục **"KỊCH BẢN BAY TỰ HÀNH (ORCA & APF)"**, bấm các nút để chạy kịch bản:
   - **`Show 30 Năm (ORCA - Minimum Snap)`**: Bay 6 pha biểu diễn hình chữ số 3, 0 và Ngôi sao với quỹ đạo tối ưu mượt mà.
   - **`Show 30 Năm (ORCA - B-Spline)`**: Bay biểu diễn sử dụng B-Spline làm mượt.
   - **`Show 30 Năm (APF Smooth)`**: Kịch bản trường thế nhân tạo APF nội suy S-Curve.
   - **`Bầy Đàn APF Leader-Follower`**: Đội hình bám đuôi Leader né tránh va chạm 3D.
   - **`Đổi Chỗ Đối Đầu (Chống Đi Xuyên)`**: Drone 2 và Drone 3 lượn chữ S đổi chỗ đối đầu.

---

## 🌐 CÁCH 2: CHẠY TRỰC QUAN HOÁ 3D VỚI GAZEBO HARMONIC (WSL2)
*Hỗ trợ sẵn Gazebo Harmonic (gz-harmonic) trên WSL2 Ubuntu 24.04 kết hợp đồ hoạ WSLg của Windows 11.*

### Bước 1: Khởi động 5 Drone ảo SITL (trên Windows)
Mở Terminal PowerShell trên Windows:
```bash
python simulation/start_swarm_sitl.py
```

### Bước 2: Khởi động GCS Điều Khiển (trên Windows)
Nhấp đúp chuột vào file:
- **`RUN_SWARM_GCS.bat`** (hoặc `RUN_ORCA_GCS.bat`)

### Bước 3: Khởi động 3D Gazebo Harmonic Swarm Viewer (WSL2)
Nhấp đúp chuột vào:
- **`RUN_GAZEBO_SWARM.bat`** (hoặc từ WSL2: `bash simulation/run_gazebo_swarm.sh`)

Script sẽ tự động:
- Khởi động cửa sổ 3D Gazebo Sim trực tiếp lên màn hình Windows 11 qua WSLg.
- Nạp thế giới mô phỏng `simulation/gazebo_swarm_world.sdf` gồm 5 Drone FPV Kamikaze 3D với 5 màu nhận diện.
- Kết nối MAVLink đọc toạ độ 5 drone từ SITL và đồng bộ vị trí 3D mượt mà tại tần số 30 FPS.

---

## 🖥️ CÁCH 3: CHẠY KẾT HỢP VỚI RVIZ2 (NẾU CÓ ROS 2 TRÊN WSL2)
*Dành cho trường hợp đã cài đặt ROS 2 Jazzy trên WSL2 Ubuntu 24.04.*
```bash
bash simulation/wsl_start_mavros_rviz.sh
```

---

## ⚡ CÁCH 4: CHẠY TRỰC TIẾP KỊCH BẢN TỪ DÒNG LỆNH (CLI)
Nếu không dùng GCS GUI, bạn có thể chạy thẳng file kịch bản độc lập qua terminal (sau khi đã bật SITL):
```bash
# Chạy Show 30 năm ORCA Minimum Snap:
python orca_swarm/show_30years_orca.py --sitl --traj min_snap

# Chạy Show 30 năm ORCA B-Spline:
python orca_swarm/show_30years_orca.py --sitl --traj bspline

# Chạy Show 30 năm APF Smooth:
python missions/show_30years_smooth.py --sitl

# Chạy APF Leader-Follower:
python missions/swarm_apf_leader_follower.py --sitl
```
