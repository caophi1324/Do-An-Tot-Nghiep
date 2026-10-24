# 🎯 AI Face Studio — Nhận Diện Khuôn Mặt & Bám Bắt Đối Tượng Thời Gian Thực

> Hệ thống AI nhận diện khuôn mặt và theo dõi đối tượng real-time, tích hợp giao diện desktop phong cách **OBS Studio** (PySide6) và dịch vụ headless/web streaming (FastAPI).

---

## 📐 Kiến trúc Luồng Xử Lý (Pipeline)

```
Input (Webcam / DroidCam / RTSP / Video)
    │
    ▼
[YOLOv11n] — Phát hiện Người (class 0)
    │
    ▼
[ByteTrack] — Gán Track ID, bám ổn định dù quay lưng / che mặt
    │
    ▼
[RetinaFace] — Phát hiện Khuôn Mặt trong crop 60% thân trên → bbox + 5 landmark
    │
    ▼
[Landmark Filter] — Lọc góc: Yaw ≤ 45°, Roll ≤ 35°, kích thước ≥ 36px, conf ≥ 0.50
    │
    ▼
[eDifFIQA-T] — Đánh giá chất lượng ảnh khuôn mặt (0.0 → 1.0), ngưỡng ≥ 0.30
    │
    ▼
[EdgeFace] — Align 5-landmark → 112×112 → Embedding 512 chiều (L2-normalized)
    │
    ▼
[FAISS] — So khớp vector cosine similarity, ngưỡng ≥ 0.50
    │
    ▼
[Voting M/N + Hysteresis] — Ổn định danh tính: vote_min=2 trong cửa sổ 5 frames
    │
    ▼
UI Preview + Web MJPEG Stream + SQLite Events Log
```

---

## ✨ Tính Năng Chính

| Tính năng | Mô tả |
|:---|:---|
| 🎯 **Click-to-Lock Target** | Click vào người bất kỳ → hệ thống tự lấy mẫu, tạo ID, khóa và theo dõi liên tục |
| 🤖 **5-Mẫu Tự Học (Active Learning)** | Sau lock, tự động thu thập 1 Anchor + 4 góc khác nhau để nhận diện mạnh mẽ hơn |
| 🔄 **Nhận Diện Lại Nhanh** | Bước ra khỏi camera rồi vào lại → nhận diện trong ≤ 2 frames (~0.1s) |
| 📷 **Multi-Source** | Webcam USB, DroidCam (HTTP), IP Camera (RTSP), Video file, Màn hình |
| 🧈 **Mượt Tuyệt Đối** | One Euro Filter trên bbox (cx,cy,w,h) — không rung giật dù đứng yên |
| 📊 **HUD Tags Real-time** | Thẻ mục tiêu (avatar + ID + quality + progress) hiển thị ở cạnh màn hình |
| ⚙️ **Tuning Thời Gian Thực** | Điều chỉnh ngưỡng nhận diện, quality, smoothing mà không cần restart |
| 💾 **Lưu Cài Đặt Tự Động** | Mọi thay đổi được lưu ngay, tắt/bật app cài đặt vẫn giữ nguyên |
| 🌐 **Web Dashboard** | Xem stream và quản lý qua trình duyệt (FastAPI + MJPEG) |

---

## 🚀 Cài Đặt & Khởi Động

### Yêu Cầu Hệ Thống

| Thành phần | Tối thiểu | Khuyến nghị |
|:---|:---|:---|
| **Python** | 3.10+ | 3.11 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | — (CPU mode) | AMD RX 5000+ (DirectML) / NVIDIA (CUDA) |
| **OS** | Windows 10+ | Windows 11 (đã test đầy đủ) |
| **Webcam** | USB 720p | USB/IP 1080p hoặc DroidCam |

### Bước 1 — Kích Hoạt Môi Trường

```powershell
# Kích hoạt venv đi kèm dự án
.\.venv\Scripts\Activate.ps1

# Kiểm tra Python
python --version  # Cần 3.10+
```

### Bước 2 — Cài Đặt Thư Viện

```powershell
pip install -e ".[dev]"
```

#### ⚡ Tăng Tốc GPU: NVIDIA (RTX A2000 4GB, RTX 30xx/40xx) & AMD

Dự án hỗ trợ **cả 2 cơ chế tăng tốc GPU** cho người dùng NVIDIA:

| Cơ chế | Hỗ trợ GPU | Cài đặt | Ghi chú |
|:---|:---|:---|:---|
| **DirectML (Mặc định)** | NVIDIA, AMD, Intel | Đã có sẵn trong venv | Tận dụng DirectX 12. **Chạy ngay trên RTX A2000 4GB** mà không cần cài CUDA Toolkit/cuDNN. |
| **Native NVIDIA CUDA** | NVIDIA RTX / GTX | `pip uninstall onnxruntime-directml`<br>`pip install onnxruntime-gpu` | Chạy trực tiếp qua CUDA & Tensor Cores. Yêu cầu máy đã cài CUDA 11.8/12.x + cuDNN. |

> **💡 Bộ nhớ VRAM (Card 4GB như RTX A2000):**
> Toàn bộ 4 mô hình AI (YOLOv11n: ~150MB, RetinaFace: ~50MB, EdgeFace: ~70MB, eDifFIQA: ~80MB) chỉ tiêu thụ tổng cộng **~350–500 MB VRAM** khi chạy. Card RTX A2000 4GB hoàn toàn dư sức đáp ứng mượt mà (dư hơn 3.5GB VRAM), không bao giờ lo tràn bộ nhớ (OOM).

### Bước 3 — Chạy Ứng Dụng

```powershell
python ui/main.py
```

> **Lần đầu chạy:** Models sẽ được tải tự động từ cache (uniface, EdgeFace, eDifFIQA). Quá trình này mất 10-30 giây.
> Thanh trạng thái bên dưới sẽ tự động hiển thị card đồ họa: `GPU: DirectML (NVIDIA RTX A2000 4GB)` hoặc `GPU: CUDA (NVIDIA RTX A2000 4GB)`.

---

## 🖥️ Hướng Dẫn Sử Dụng Giao Diện

### Bố Cục Màn Hình

```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 AI Face Studio                              FPS 28 | CPU 45%│
├───────────┬─────────────────────────────────┬───────────────────┤
│  Target   │                                 │  Modes Dock       │
│  Card     │     📹 Video Preview            │  Sources Dock     │
│  (trái)   │     (khu vực chính)             │  Tuning Dock      │
│           │                                 │  Overlay Dock     │
│  Scene    │                                 │  Controls Dock    │
│  Roster   │                                 │                   │
│  (phải)   │                                 │                   │
├───────────┴─────────────────────────────────┴───────────────────┤
│  Status Bar: Track 1 | FPS 28 | CPU 45% | GPU DirectML | 42ms  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Hướng Dẫn Sử Dụng Từng Bước

### 🎬 1. Thêm Nguồn Video

1. Mở **Dock Nguồn (Sources)** ở cột phải
2. Nhấn **➕ Thêm nguồn** → chọn loại:
   - **Webcam USB**: Chọn Device ID (0 = camera đầu tiên)
   - **DroidCam**: Nhập IP:Port (vd: `192.168.1.100:4747`) — hệ thống tự thêm `/video`
   - **RTSP Camera**: Nhập URL đầy đủ `rtsp://...`
   - **Video File**: Chọn file `.mp4`, `.avi`, `.mkv`

### ▶️ 2. Bắt Đầu Pipeline

- Nhấn **▶ BẮT ĐẦU** trong **Dock Điều khiển** hoặc nhấn `Space`
- Camera sẽ hiển thị video với bounding box màu **xanh dương** (Unknown) xung quanh các người phát hiện được

### 🎯 3. Click Chọn & Khóa Mục Tiêu (Click-to-Lock)

1. **Click chuột trái** vào người bạn muốn theo dõi trong khung hình
2. Hệ thống tự động:
   - 📸 Lấy mẫu khuôn mặt ngay lập tức
   - 🔑 Gán ID mới (Person 1, Person 2, ...)
   - 🔒 Khóa mục tiêu → viền bbox chuyển màu **vàng gold** + góc reticle
   - Hiển thị badge `🎯 ADAPTING 1/5`
3. **Giữ nguyên vị trí** trong 2-3 giây, sau đó **từ từ xoay đầu** nhẹ nhàng sang trái, phải, cúi/ngẩng để hệ thống thu thập 5 mẫu góc khác nhau
4. Khi đạt `Ready (5/5)` → nhận diện mạnh mẽ ở mọi góc độ

> **💡 Mẹo:** Nếu click vào người đã từng đăng ký → hệ thống nhận ra và liên kết lại ID cũ, **không tạo trùng**.

### 🔓 4. Mở Khóa

- **Click vào người đang lock** → Mở khóa (Unlock)
- **Click vào vùng trống** → Mở khóa tất cả

### 🧑‍💼 5. Đăng Ký Thủ Công (Enrollment Dialog)

Dùng khi muốn đặt tên cụ thể thay vì "Person N":

1. Nhấn `E` hoặc click **Đăng ký** trong Controls Dock
2. Cửa sổ đăng ký mở ra với camera live
3. Nhập Tên, Mã, Nhóm
4. Nhìn thẳng camera → hệ thống tự chọn ảnh chất lượng tốt nhất
5. Nhấn **Lưu** để hoàn tất

---

## ⚙️ Tinh Chỉnh Nâng Cao (Tuning Dock)

| Tham số | Ý nghĩa | Khi nào điều chỉnh |
|:---|:---|:---|
| **Ngưỡng nhận diện** (Match Thr) | Cosine similarity để chấp nhận là "đã biết" | Giảm nếu hay bị Unknown; Tăng nếu nhận nhầm người |
| **Quality tối thiểu** (Min Quality) | Chất lượng eDifFIQA tối thiểu (0-100%) | Giảm nếu ít lấy mẫu được; Tăng nếu ảnh quá mờ vẫn nhận |
| **Độ mượt Cutoff** | OneEuro filter cutoff — cao = mượt hơn | Tăng nếu bbox rung; Giảm nếu muốn phản hồi nhanh |
| **Độ nhạy Beta** | OneEuro filter beta — cao = nhạy chuyển động hơn | Tăng nếu tracking trễ khi di chuyển nhanh |
| **Góc ngang Yaw** | Góc quay đầu tối đa chấp nhận (0-80%) | Tăng nếu muốn nhận cả khi quay nghiêng nhiều |

### Cài Đặt Mặc Định Khuyến Nghị

| Kịch bản | Match Thr | Quality | Yaw |
|:---|:---|:---|:---|
| Ngồi cố định trước cam | 0.50 | 30% | 45% |
| Di chuyển / Drone | 0.45 | 20% | 60% |
| Nhiều người trong phòng | 0.60 | 35% | 40% |

---

## 🗑️ Xóa & Reset Dữ Liệu

### Xóa Toàn Bộ Database

1. Nhấn **🗑️ Xóa cơ sở dữ liệu** trong Controls Dock
2. Xác nhận trong hộp thoại
3. **Kết quả:**
   - Tất cả ID người → xóa hoàn toàn khỏi SQLite + FAISS
   - Track hiện tại → chuyển về Unknown
   - Avatar, thumbnails → reset
   - Lần click tiếp → bắt đầu lại từ **Person 1 (ID 1)**

> **⚠️ Lưu ý:** Thao tác này KHÔNG THỂ HOÀN TÁC. Hãy chắc chắn trước khi xóa.

### Xóa Từng Người

Hiện tại tính năng xóa từng người khả dụng qua **Cửa sổ Đăng ký** → chọn người → nhấn **Xóa**.

---

## ⌨️ Phím Tắt

| Phím | Chức năng |
|:---|:---|
| `Space` | Bắt đầu / Dừng pipeline |
| `E` | Mở cửa sổ Đăng ký khuôn mặt |
| `F11` | Fullscreen / Thoát fullscreen |
| `Click chuột trái` | Click-to-Lock / Unlock mục tiêu |

---

## 📺 DroidCam — Dùng Điện Thoại Làm Camera

1. Cài app **DroidCam** trên điện thoại Android/iOS
2. Kết nối điện thoại và PC cùng WiFi
3. Mở app DroidCam → ghi nhớ địa chỉ IP (vd: `192.168.1.5`)
4. Trong ứng dụng → **Sources Dock** → **Thêm nguồn** → chọn **Stream (HTTP/RTSP)**
5. Nhập: `192.168.1.5:4747` (hệ thống tự thêm `/video`)
6. Nhấn **Kết nối** → ▶ **Bắt đầu**

> **Nếu không kết nối được:** Tắt firewall Windows tạm thời, kiểm tra cùng mạng WiFi.

---

## 🌐 Web Dashboard (FastAPI)

Chạy dịch vụ headless để xem qua browser hoặc tích hợp API:

```powershell
# Chạy web service
uvicorn service.main:app --host 0.0.0.0 --port 8000

# Xem stream tại:
# http://localhost:8000              — Dashboard
# http://localhost:8000/stream.mjpg — Live MJPEG stream
# http://localhost:8000/docs        — API docs (Swagger)
```

### Cấu Hình `.env`

```bash
cp .env.example .env
# Chỉnh sửa .env:
JWT_SECRET=<chuỗi 32 ký tự ngẫu nhiên>   # BẮT BUỘC
ADMIN_PASSWORD=<mật_khẩu>                 # hoặc để trống = tự sinh
DATABASE_URL=sqlite:///data/face_app.db
```

---

## 🏗️ Cấu Trúc Thư Mục

```
Branch_Aitracking_PersonalUSE/
├── core/                    # Lõi xử lý (độc lập với UI)
│   ├── config.py            # Cấu hình & Presets
│   ├── detectors.py         # YOLOv11n + RetinaFace
│   ├── tracker.py           # ByteTrack + TrackState + Voting + FaceGallery
│   ├── quality.py           # Filter 5-landmark + eDifFIQA-T
│   ├── recognizer.py        # Align + EdgeFace + FAISS
│   ├── smoothing.py         # One Euro Filter
│   ├── sources.py           # Webcam, DroidCam, RTSP, Video
│   ├── runner.py            # PipelineRunner (background thread)
│   └── pipeline.py          # Master Pipeline orchestrator
├── db/                      # Cơ sở dữ liệu
│   ├── models.py            # SQLAlchemy ORM
│   └── repo.py              # CRUD + FAISS sync + Events
├── ui/                      # Giao diện Desktop PySide6
│   ├── main_window.py       # Cửa sổ chính (OBS-style)
│   ├── video_widget.py      # Video display + HUD Tags
│   ├── theme.qss            # Dark Theme
│   ├── docks/               # Modes, Sources, Tuning, Overlay, Controls
│   ├── dialogs/             # Enrollment, Settings, Events History
│   └── main.py              # Entry point
├── service/                 # FastAPI Web Service
│   ├── main.py              # REST API + Web Dashboard
│   └── auth.py              # JWT Authentication
├── tests/                   # Test suite (67 passed ✅)
├── data/                    # DB, Snapshots, FAISS Index, Recordings
├── models/                  # ONNX model files
└── pyproject.toml
```

---

## 🔍 Câu Hỏi Thường Gặp (FAQ)

**Q: Ngưỡng Quality là bao nhiêu để khuôn mặt được nhận?**
> A: Mặc định **Quality ≥ 0.30 (30%)** theo mô hình eDifFIQA-T. Đây là điểm sàn chất lượng — thấp hơn có thể ảnh mờ/nghiêng quá nhiều.

**Q: Sao click mãi vẫn 0/5 mẫu?**
> A: Hãy đảm bảo khuôn mặt **nhìn thẳng camera**, đủ ánh sáng, không bị che. Badge `ADAPTING 1/5` xuất hiện ngay khi lấy được mẫu đầu tiên. Nếu vẫn 0/5 → thử giảm Quality xuống 20% trong Tuning Dock.

**Q: Bước ra ngoài camera rồi vào lại không nhận ra?**
> A: Sau khi đạt `Ready (5/5)`, nhận diện lại hoạt động trong ~2 frames. Nếu vẫn không nhận ra:
> - Ánh sáng thay đổi quá nhiều → thử giảm Match Thr xuống 0.45
> - Chưa đủ 5 mẫu → cần nhận đủ mẫu trước

**Q: Nhận diện nhầm người lạ là tôi?**
> A: Tăng **Match Thr** lên 0.60-0.65 trong Tuning Dock.

**Q: DroidCam báo lỗi kết nối?**
> A: Kiểm tra: (1) Cùng mạng WiFi, (2) Nhập đúng IP:4747 (không cần thêm `/video`), (3) Tắt tạm firewall Windows.

**Q: Cài đặt không lưu khi tắt app?**
> A: Đã được fix — mọi thay đổi tự động lưu vào `data/config.json` khi tắt. Nếu vẫn không lưu, kiểm tra quyền ghi thư mục `data/`.

---

## 🧪 Chạy Tests

```powershell
.\.venv\Scripts\pytest.exe tests -v
# Expected: 67 passed, 3 xfailed, 0 failed
```

---

## 🐳 Triển Khai Docker (Web Service)

```bash
cp .env.example .env
# Thiết lập JWT_SECRET trong .env
docker compose up -d --build
# Truy cập http://<IP_MÁY>:8000
```

---

## 📊 Kết Quả Kiểm Thử (Test Report)

| Test Suite | Số test | Kết quả |
|:---|:---:|:---:|
| `test_bugs.py` | 34 | ✅ 31 passed, 3 xfailed |
| `test_integration.py` | 13 | ✅ 13 passed |
| `test_unit.py` | 23 | ✅ 23 passed |
| **Tổng** | **70** | **✅ 67 passed, 3 xfailed** |

> *3 xfailed là các bug đã biết và chưa ưu tiên fix (re-enroll template, path traversal trong code, Settings dialog storage path).*

---

## 📝 Ghi Chú Kỹ Thuật

- **Mô hình nhận diện:** EdgeFace-XXS (512-dim, ~2MB, chạy trên CPU/DirectML)
- **Mô hình phát hiện mặt:** RetinaFace MobileNet v2 (ONNX)
- **Mô hình phát hiện người:** YOLOv11n (ONNX, 6MB)
- **Chất lượng ảnh:** eDifFIQA-T (ONNX)
- **Vector store:** FAISS IndexFlatIP (Inner Product = Cosine trên L2-normalized vectors)
- **Hardware acceleration:** DirectML (AMD/Intel), CUDA (NVIDIA), CPU fallback
