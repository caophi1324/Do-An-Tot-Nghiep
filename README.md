# Đồ Án Tốt Nghiệp

Workspace tổng hợp gồm 3 mảng chính: **phần cứng flight controller**, **mô phỏng bầy drone (SITL)** và **hệ thống nhận diện & bám sát khuôn mặt bằng AI**.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `Drone/` | Thiết kế phần cứng: flight controller `Drone ver2` (KiCad, BOM kèm LCSC), tay cầm điều khiển `Tay cam` (Altium), script tự động layout PCB (`Lib Kicad/arrange_*.py`), và 2 dự án tham khảo clone từ GitHub (`ESC_4in1_G071_1S3_SIZ200`, `Open-ESPilot`) |
| `drone_swarm_simulation_win11_wsl/` | Mô phỏng bầy drone ArduPilot SITL trên WSL: ứng dụng GCS (PySide6), engine ORCA/RVO2 + APF, điều khiển bằng gamepad, tích hợp Gazebo/RViz2. Hướng dẫn chạy: `HD_CHAY_MOPHONG_WIN11_WSL.md` |
| `StackMod/` | AI Face Tracking Studio: pipeline nhận diện khuôn mặt (YOLO + RetinaFace + EdgeFace + eDifFIQA), UI PySide6, REST API FastAPI, Docker. Hướng dẫn: `StackMod/README.md` |
| `fimware skill/` | Bộ 8 skill `.skill` (hướng dẫn vận hành drone: bringup, flight-safety, log-review, mavlink-review, sitl-sim, tuning...) kèm manifest SHA256 |

## Nội dung KHÔNG đưa lên repo (giữ trên máy local)

- `Drone/LIb AL/` — thư viện Altium cá nhân, các file >100 MB vượt giới hạn GitHub.
- `Drone/Lib Kicad/kicad-footprints`, `kicad-symbols-master` — thư viện chuẩn KiCad, cài lại qua KiCad hoặc clone từ [KiCad gitlab](https://gitlab.com/kicad).
- `drone_swarm_simulation_win11_wsl/ardupilot_wiki/` — clone wiki ArduPilot (~2.2 GB), xem [ArduPilot/ardupilot_wiki](https://github.com/ArduPilot/ardupilot_wiki).
- `StackMod/data/` — database + ảnh snapshots sinh khi chạy (chứa dữ liệu khuôn mặt).
- `StackMod/.venv/` — môi trường ảo Python.
- Các file `.zip`/`.rar` — bản nén trùng với nội dung đã giải nén.

## Ghi chú về các repo git cũ (`.git.bak`)

Các thư mục con trước đây có repo git riêng (`StackMod`, `Drone/ESC_4in1_G071_1S3_SIZ200`, `Drone/Open-ESPilot`, `Drone/Tay cam`) đã được gộp nội dung vào repo này; thư mục `.git` cũ được đổi tên thành `.git.bak` (bị gitignore). Muốn khôi phục repo riêng cũ, đổi tên `.git.bak` trở lại thành `.git`.

## Cài đặt nhanh StackMod

```bash
cd StackMod
python -m venv .venv && .venv\Scripts\activate
pip install -e .
python tools/download_models.py   # nếu chưa có models/*.onnx
```
