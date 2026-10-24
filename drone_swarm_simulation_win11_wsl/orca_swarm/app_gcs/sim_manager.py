#!/usr/bin/env python3
"""
DRONE SWARM GCS - SIMULATION PROCESS MANAGER
Quản lý các tiến trình mô phỏng Swarm SITL, ROS 2 Mesh Publisher và RViz2.
Cung cấp điều khiển 1-click trực tiếp trên GUI mà không cần mở nhiều cửa sổ terminal.
"""

import os
import sys
import time
import subprocess
import threading
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


class SwarmSimulationManager(QObject):
    """
    Quản lý vòng đời của 3 tiến trình mô phỏng:
    1. SITL 5 Drone (ArduCopter.exe)
    2. ROS 2 3D Mesh Publisher (drone_3d_mesh_publisher.py)
    3. RViz2 3D Visualization GUI (rviz2.exe)
    """
    # Signals phát trạng thái: (sitl_active, publisher_active, rviz_active)
    status_updated = pyqtSignal(bool, bool, bool)
    log_emitted = pyqtSignal(str, str)  # (level, message)

    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.project_root = os.path.abspath(project_root)
        self.sitl_proc: Optional[subprocess.Popen] = None
        self.pub_proc: Optional[subprocess.Popen] = None
        self.rviz_proc: Optional[subprocess.Popen] = None

        self.conda_bat = r"C:\ProgramData\miniforge3\condabin\conda.bat"
        self.python_exe = sys.executable

        # Timer định kỳ 1.0s kiểm tra trạng thái tiến trình
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.poll_status)
        self.monitor_timer.start(1000)

    def is_sitl_running(self) -> bool:
        if self.sitl_proc is not None and self.sitl_proc.poll() is None:
            return True
        return self._is_process_name_running("ArduCopter.exe")

    def is_publisher_running(self) -> bool:
        return self.pub_proc is not None and self.pub_proc.poll() is None

    def is_rviz_running(self) -> bool:
        if self.rviz_proc is not None and self.rviz_proc.poll() is None:
            return True
        if self._is_process_name_running("rviz2.exe") or self._is_process_name_running("gz.exe"):
            return True
        try:
            out = subprocess.check_output('wsl pgrep -f "gz sim"', shell=True, text=True, stderr=subprocess.DEVNULL)
            return len(out.strip()) > 0
        except Exception:
            return False

    def _is_process_name_running(self, proc_name: str) -> bool:
        try:
            cmd = f'tasklist /FI "IMAGENAME eq {proc_name}" /NH'
            out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            return proc_name.lower() in out.lower()
        except Exception:
            return False

    def poll_status(self):
        """Phát signal cập nhật trạng thái tiến trình lên giao diện."""
        s_sitl = self.is_sitl_running()
        s_pub = self.is_publisher_running()
        s_rviz = self.is_rviz_running()
        self.status_updated.emit(s_sitl, s_pub, s_rviz)

    def start_sitl(self, num_drones: int = 3) -> bool:
        """Khởi chạy script tạo drone SITL ArduCopter."""
        if self.is_sitl_running():
            self.log_emitted.emit("WARNING", f"SITL {num_drones} Drone đã đang chạy!")
            return True

        sitl_script = os.path.join(self.project_root, "simulation", "start_swarm_sitl.py")
        if not os.path.exists(sitl_script):
            self.log_emitted.emit("ERROR", f"Không tìm thấy file {sitl_script}")
            return False

        try:
            self.log_emitted.emit("INFO", f"Đang khởi động SITL {num_drones} Drones (ArduCopter)...")
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            self.sitl_proc = subprocess.Popen(
                [self.python_exe, sitl_script, str(num_drones)],
                cwd=self.project_root,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.log_emitted.emit("SUCCESS", f"SITL {num_drones} Drone đã khởi chạy thành công (PID: {self.sitl_proc.pid})")
            self.poll_status()
            return True
        except Exception as e:
            self.log_emitted.emit("ERROR", f"Khởi động SITL thất bại: {e}")
            return False

    def start_mesh_publisher(self, num_drones: int = 3) -> bool:
        """Khởi chạy node ROS 2 chuyển tiếp 3D Model & Telemetry sang RViz2."""
        if self.is_publisher_running():
            self.log_emitted.emit("WARNING", "3D Mesh Publisher đã đang chạy!")
            return True

        script_path = os.path.join(self.project_root, "visualization", "drone_3d_mesh_publisher.py")
        if not os.path.exists(script_path):
            self.log_emitted.emit("ERROR", f"Không tìm thấy file {script_path}")
            return False

        try:
            self.log_emitted.emit("INFO", "Đang khởi động 3D Mesh Publisher qua Conda ros2_win...")
            cmd = f'cmd /c "set ROS_LOCALHOST_ONLY=1 && {self.conda_bat} activate ros2_win && python visualization\\drone_3d_mesh_publisher.py {num_drones}"'

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            self.pub_proc = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                shell=True,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.log_emitted.emit("SUCCESS", f"3D Mesh Publisher đã khởi chạy (PID: {self.pub_proc.pid})")
            self.poll_status()
            return True
        except Exception as e:
            self.log_emitted.emit("ERROR", f"Khởi động 3D Mesh Publisher thất bại: {e}")
            return False

    def start_rviz(self, num_drones: int = 3) -> bool:
        """Mở cửa sổ giao diện 3D (Gazebo Harmonic qua WSL2 hoặc RViz2)."""
        if self.is_rviz_running():
            self.log_emitted.emit("WARNING", "Giao diện 3D Viewer đã đang mở!")
            return True

        try:
            if os.path.exists(self.conda_bat):
                self.log_emitted.emit("INFO", "Đang mở cửa sổ RViz2 3D Viewer...")
                cmd = f'cmd /c "set ROS_LOCALHOST_ONLY=1 && {self.conda_bat} activate ros2_win && rviz2 -d config\\swarm_view.rviz"'
            else:
                self.log_emitted.emit("INFO", f"Đang mở giao diện 3D Gazebo Harmonic ({num_drones} Drones) qua WSL2...")
                cmd = f'wsl bash simulation/run_gazebo_swarm.sh {num_drones}'

            self.rviz_proc = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.log_emitted.emit("SUCCESS", "Đã mở thành công giao diện 3D Viewer.")
            self.poll_status()
            return True
        except Exception as e:
            self.log_emitted.emit("ERROR", f"Mở 3D Viewer thất bại: {e}")
            return False

    def start_all_simulation(self, num_drones: int = 3):
        """Khởi động toàn bộ chuỗi mô phỏng trong một luồng riêng để không block GUI."""
        def _worker():
            self.log_emitted.emit("INFO", "=== BẮT ĐẦU CHUỖI KHỞI ĐỘNG TOÀN BỘ MÔ PHỎNG (1-CLICK) ===")
            # 1. Bật SITL
            if not self.is_sitl_running():
                self.start_sitl(num_drones)
                self.log_emitted.emit("INFO", f"Đang đợi {num_drones} Drone SITL khởi tạo cổng mạng (khoảng 8s)...")
                time.sleep(8.0)

            # 2. Bật Mesh Publisher (nếu có Conda ROS 2)
            if os.path.exists(self.conda_bat) and not self.is_publisher_running():
                self.start_mesh_publisher(num_drones)
                time.sleep(2.0)

            # 3. Mở 3D Viewer (Gazebo Harmonic trên WSL2 hoặc RViz2)
            if not self.is_rviz_running():
                self.start_rviz(num_drones)

            self.log_emitted.emit("SUCCESS", "=== ĐÃ KHỞI ĐỘNG TOÀN BỘ HỆ THỐNG MÔ PHỎNG THÀNH CÔNG ===")

        threading.Thread(target=_worker, daemon=True).start()

    def stop_all_simulation(self):
        """Dọn dẹp sạch sẽ toàn bộ các tiến trình mô phỏng và giải phóng cổng."""
        def _worker():
            self.log_emitted.emit("WARNING", "Đang dừng toàn bộ các tiến trình mô phỏng...")

            # 1. Kill tiến trình RViz / Gazebo
            if self.rviz_proc:
                try:
                    subprocess.run(f"taskkill /F /T /PID {self.rviz_proc.pid}", shell=True, capture_output=True)
                except Exception:
                    pass
                self.rviz_proc = None

            # 2. Kill tiến trình Publisher
            if self.pub_proc:
                try:
                    subprocess.run(f"taskkill /F /T /PID {self.pub_proc.pid}", shell=True, capture_output=True)
                except Exception:
                    pass
                self.pub_proc = None

            # 3. Kill tiến trình SITL script
            if self.sitl_proc:
                try:
                    subprocess.run(f"taskkill /F /T /PID {self.sitl_proc.pid}", shell=True, capture_output=True)
                except Exception:
                    pass
                self.sitl_proc = None

            # 4. Quét dọn các process con mồ côi (ArduCopter.exe, rviz2.exe, Gazebo WSL)
            subprocess.run("taskkill /F /IM ArduCopter.exe", shell=True, capture_output=True)
            subprocess.run("taskkill /F /IM rviz2.exe", shell=True, capture_output=True)
            subprocess.run("wsl bash -c 'killall -9 arducopter ArduCopter gz \"gz sim\" 2>/dev/null; pkill -9 -f wsl_gazebo_swarm_bridge 2>/dev/null || true'", shell=True, capture_output=True)

            time.sleep(1.0)
            self.log_emitted.emit("SUCCESS", "Đã tắt toàn bộ mô phỏng và giải phóng tất cả cổng kết nối.")
            self.poll_status()

        threading.Thread(target=_worker, daemon=True).start()
