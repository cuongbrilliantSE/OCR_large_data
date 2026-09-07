# HƯỚNG DẪN DEPLOY PRODUCTION - OCR LARGE DATA PIPELINE

Tài liệu này hướng dẫn 2 phương án đưa dự án lên máy chủ Production ổn định, bảo mật và chạy 24/7.

---

## 🚀 PHƯƠNG ÁN 1: Deploy bằng Docker & Docker Compose (Khuyên dùng)

### Yêu cầu tiên quyết:
- Máy chủ đã cài đặt `docker` và `docker compose`.
- Đã clone mã nguồn dự án vào thư mục trên server.

### Các bước thực hiện:

1. **Cấu hình biến môi trường (`.env`):**
   ```bash
   cp .env.example .env # hoặc chỉnh sửa file .env
   nano .env
   ```
   Điền khóa API Gemini của bạn:
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```

2. **Khởi chạy container ở chế độ chạy nền (detached):**
   ```bash
   docker compose up -d --build
   ```

3. **Kiểm tra trạng thái container:**
   ```bash
   docker compose ps
   docker compose logs -f
   ```

4. **Dữ liệu bền vững (Persistence):**
   - File cơ sở dữ liệu và kết quả OCR được map vào thư mục `./data` trên máy chủ, đảm bảo không bị mất dữ liệu khi restart hoặc rebuild container.

---

## 🐧 PHƯƠNG ÁN 2: Deploy trực tiếp trên Linux VPS (Ubuntu / Debian)

### 1. Cài đặt các gói hệ thống cần thiết
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv p7zip-full p7zip-rar nginx certbot python3-certbot-nginx
```

### 2. Thiết lập mã nguồn & Virtual Environment
```bash
# Ví dụ thư mục dự án tại /var/www/OCR_large_data
cd /var/www/OCR_large_data

# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Phân quyền thư mục lưu trữ
```bash
sudo chown -R www-data:www-data /var/www/OCR_large_data/data
sudo chown -R www-data:www-data /var/www/OCR_large_data/logs
sudo chmod -R 775 /var/www/OCR_large_data/data
sudo chmod -R 775 /var/www/OCR_large_data/logs
```

### 4. Cấu hình Systemd Service (Tự khởi động cùng hệ điều hành)
Copy file cấu hình service:
```bash
sudo cp deploy/ocr-studio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ocr-studio
sudo systemctl start ocr-studio
```
Kiểm tra trạng thái service:
```bash
sudo systemctl status ocr-studio
```

### 5. Cấu hình Nginx Reverse Proxy
```bash
sudo cp deploy/nginx-ocr.conf /etc/nginx/sites-available/ocr-studio
sudo ln -s /etc/nginx/sites-available/ocr-studio /etc/nginx/sites-enabled/
```
*Lưu ý: Mở file `/etc/nginx/sites-available/ocr-studio` và thay `your_domain.com` bằng tên miền hoặc IP máy chủ của bạn.*

Kiểm tra cú pháp và kích hoạt:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Cài đặt SSL miễn phí với Let's Encrypt (HTTPS)
```bash
sudo certbot --nginx -d your_domain.com
```

---

## ⚙️ CHECKLIST BẢO MẬT & TỐI ƯU TRÊN PRODUCTION

1. **Giới hạn dung lượng tải lên (Upload Body Size):**
   - Đã cấu hình `client_max_body_size 1024M;` trong Nginx để người dùng có thể tải lên các file RAR/ZIP sách scan nặng hàng trăm MB đến 1GB.
2. **Timeout xử lý (Proxy Timeouts):**
   - Đã tăng `proxy_read_timeout 600s;` tránh trường hợp Nginx ngắt kết nối khi đang giải nén file sách lớn.
3. **Tường lửa (UFW):**
   - Chỉ mở port 80, 443 và SSH (22):
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw allow OpenSSH
   sudo ufw enable
   ```
