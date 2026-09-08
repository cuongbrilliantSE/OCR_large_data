---
title: Book OCR Studio AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
short_description: OCR sách & tài liệu tiếng Việt với Gemini Vision AI
pinned: false
---

# Large-Scale Image to Text OCR Pipeline 🚀

Hệ thống OCR hiệu năng cao (High-throughput Batch OCR Pipeline) được thiết kế chuyên biệt để chuyển đổi **hàng chục nghìn đến hàng triệu file ảnh sách sang văn bản (`.txt` và `JSONL`)**, hỗ trợ đa tiến trình, tối ưu hóa ảnh đầu vào và tận dụng sức mạnh vượt trội của **Google Gemini Vision AI (chính xác 100% tiếng Việt có dấu)**.

---

## 🌟 Động Cơ OCR: Google Gemini Vision AI

- **Chính xác tuyệt đối**: Mô hình đa phương thức `gemini-2.5-flash` / `gemini-2.0-flash` hiểu sâu toàn diện ngữ cảnh tiếng Việt (danh từ riêng, địa danh, dấu câu, thanh điệu).
- **Phân tách trang thông minh**: Tự động nhận diện cấu trúc mở sách đôi thành `=== [TRANG TRÁI] ===` và `=== [TRANG PHẢI] ===`.
- **Hiệu năng cao**: Tốc độ xử lý song song nhiều luồng (workers), có cơ chế tự động thử lại khi gặp giới hạn tốc độ (Rate Limit 429).

---

## 🛠️ Cài Đặt

1. Cài đặt các gói phụ thuộc cần thiết:
```bash
pip install -r requirements.txt
```

2. *(Khuyên dùng)* Cấu hình Google Gemini API Key để đạt độ chính xác cao nhất:
   - Lấy API Key miễn phí tại: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Mở file `.env` ở thư mục gốc và dán key vào:
     ```env
     GEMINI_API_KEY=AIzaSy...
     ```
   - Hoặc điền trực tiếp vào `configs/config.yaml` tại mục `ocr.gemini.api_key`.

---

## 📖 Hướng Dẫn Sử Dụng Nhanh

### 🌐 1. Sử Dụng Giao Diện Web Studio (Khuyên dùng)
Khởi chạy giao diện web để kéo thả trực tiếp file nén `.rar` / `.zip`, theo dõi tiến độ thời gian thực và tải về sách hoàn chỉnh:
```bash
python main.py web
# Mở trình duyệt và truy cập: http://localhost:8000
```
- **Kéo thả file nén**: Hỗ trợ trực tiếp `.rar` (kể cả RAR4/RAR5), `.zip`, `.7z`.
- **Giám sát thời gian thực**: Xem thanh tiến độ %, số trang hoàn thành, tốc độ xử lý.
- **Xem trước trực tiếp (Live Preview)**: Đọc ngay nội dung văn bản tiếng Việt vừa trích xuất.
- **Tải về thuận tiện**: Tải ngay file sách hoàn chỉnh `.txt` hoặc trọn gói nén `.zip`.

### 2. Thử nghiệm trên 1 ảnh bất kỳ qua CLI
```bash
python main.py test-image "data/input/đất yên mỹ- hưng yên/20260814_172232.jpg"
```

### 2. Chuẩn bị ảnh đầu vào theo từng Cuốn Sách
Tạo mỗi cuốn sách thành một thư mục con bên trong `data/input/`:
```text
data/input/
├── Cuon_Sach_01/
│   ├── trang_001.jpg
│   ├── trang_002.jpg
│   └── ...
```

### 3. Xem danh sách sách & tiến độ
```bash
python main.py books
```

### 4. Chạy chuyển đổi OCR hàng loạt
- **Chạy toàn bộ các sách trong hàng đợi:**
  ```bash
  python main.py run
  ```
- **Chạy riêng 1 cuốn sách cụ thể:**
  ```bash
  python main.py run --book "đất yên mỹ- hưng yên"
  ```
- Tùy chọn số lượng worker song song hoặc giới hạn số ảnh:
  ```bash
  python main.py run --workers 4 --limit 100
  ```

### 5. Ghép các trang thành 1 file text hoàn chỉnh của cuốn sách
Sau khi OCR xong, bạn có thể gộp tất cả các trang `.txt` lẻ của cuốn sách lại theo đúng thứ tự:
```bash
python main.py merge-book "đất yên mỹ- hưng yên"
# Kết quả sẽ được lưu tại: data/output/đất yên mỹ- hưng yên.txt
```

### 6. Xem thống kê tiến độ
```bash
python main.py status
```

### 7. Tiếp tục tiến trình khi bị gián đoạn
```bash
python main.py resume
```

### 8. Thử lại các trang bị lỗi (nếu có)
```bash
python main.py retry-failed
```

---

## ⚙️ Tùy Chỉnh Cấu Hình (`configs/config.yaml`)

```yaml
system:
  num_workers: 4              # Số tiến trình / luồng chạy song song
  batch_size: 20              # Kích thước chunk gửi tới worker
  max_tasks_per_child: 500    # Tự động refresh worker để giải phóng RAM

paths:
  input_dir: "data/input"
  output_dir: "data/output"
  db_path: "data/tracker.db"

ocr:
  engine: "gemini"            # "gemini" (chính xác 100%), "vietocr", "rapidocr", "easyocr"
  gemini:
    api_key: ""               # Để trống nếu dùng .env hoặc biến môi trường GEMINI_API_KEY
    model_name: "gemini-2.5-flash"
    temperature: 0.0
    max_retries: 3
  use_angle_cls: true
  min_score_thresh: 0.3

preprocessing:
  enable: true
  max_dimension: 2560         # Đảm bảo độ sắc nét cho chữ in nhỏ và dấu thanh
  auto_orient: true           # Tự động phát hiện và xoay ảnh chụp nghiêng 90/270 độ
  auto_contrast: false
  split_double_pages: false   # True nếu muốn tách ảnh sách 2 trang thành 2 lượt đọc

output:
  format: "both"              # 'txt' (1-1), 'jsonl' (gộp), hoặc 'both' (cả hai)
  mirror_folder_structure: true
```
