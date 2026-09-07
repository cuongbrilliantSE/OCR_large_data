import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Any
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
import gdown
from gdown.parse_url import parse_url
from gdown.download_folder import _download_and_parse_google_drive_link, _get_session, _extract_folder_id

from .archive_extractor import extract_archive, VALID_IMAGE_EXTS
from src.utils.file_utils import natural_sort_key

MODERN_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def format_bytes(num_bytes: float) -> str:
    """Format bytes into human readable format (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def is_gdrive_url(url: str) -> bool:
    """Check if a string looks like a Google Drive URL."""
    if not url:
        return False
    url = url.strip()
    return "drive.google.com" in url or "docs.google.com" in url


class GDriveTaskManager:
    """Tracks asynchronous Google Drive download tasks and real-time progress."""
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        task_id = uuid.uuid4().hex[:10]
        with self._lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "percent": 0.0,
                "message": "Đang khởi tạo...",
                "current_str": "0 B",
                "total_str": "---",
                "speed": "---",
                "logs": [f"[{datetime.now().strftime('%H:%M:%S')}] Đã tiếp nhận yêu cầu tải Google Drive."],
                "book_name": None,
                "images_count": 0,
                "error": None,
                "updated_at": time.time()
            }
        return task_id

    def update_task(self, task_id: str, message: str, meta: Optional[Dict[str, Any]] = None):
        with self._lock:
            if task_id not in self.tasks:
                return
            t = self.tasks[task_id]
            t["message"] = message
            t["updated_at"] = time.time()
            if meta:
                for k, v in meta.items():
                    t[k] = v
            log_line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            if not t["logs"] or t["logs"][-1] != log_line:
                t["logs"].append(log_line)
                if len(t["logs"]) > 100:
                    t["logs"] = t["logs"][-100:]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            return dict(task) if task else None


gdrive_task_manager = GDriveTaskManager()


class DownloadProgressTracker:
    def __init__(self, callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]]):
        self.callback = callback
        self.start_time = time.time()
        self.last_update = 0.0

    def __call__(self, current: int, total: Optional[int]):
        if not self.callback:
            return
        now = time.time()
        if now - self.last_update < 0.25 and current != total:
            return
        self.last_update = now

        elapsed = now - self.start_time
        speed_bps = (current / elapsed) if elapsed > 0 else 0.0
        speed_str = f"{format_bytes(speed_bps)}/s"

        current_str = format_bytes(current)
        total_str = format_bytes(total) if total else "---"

        if total and total > 0:
            pct = round(min(90.0, (current / total) * 90.0), 1)
            msg = f"Đang tải: {current_str} / {total_str} ({pct}%) • Tốc độ: {speed_str}"
        else:
            pct = 45.0
            msg = f"Đang tải: {current_str} • Tốc độ: {speed_str}"

        self.callback(msg, {
            "percent": pct,
            "current_str": current_str,
            "total_str": total_str,
            "speed": speed_str,
            "status": "downloading"
        })


def download_gdrive_file(file_id: str, dest_path: Path, tracker: Optional[Callable[[int, Optional[int]], None]] = None) -> bool:
    """
    Downloads a single Google Drive file using direct usercontent streaming with gdown fallback.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    session = requests.Session()
    session.headers.update({"User-Agent": MODERN_USER_AGENT})
    try:
        r = session.get(direct_url, stream=True, timeout=30)
        if r.status_code == 200 and "application/json" not in r.headers.get("Content-Type", ""):
            total = int(r.headers.get("Content-Length", 0)) or None
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if tracker:
                            tracker(downloaded, total)
            if dest_path.is_file() and dest_path.stat().st_size > 0:
                return True
    except Exception:
        pass

    try:
        res = gdown.download(
            id=file_id,
            output=str(dest_path),
            quiet=True,
            user_agent=MODERN_USER_AGENT,
            use_cookies=False,
            progress=tracker
        )
        return bool(res and dest_path.is_file() and dest_path.stat().st_size > 0)
    except Exception:
        return False


def download_and_extract_gdrive(
    url: str,
    input_root: Path,
    custom_book_name: Optional[str] = None,
    status_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None
) -> Tuple[str, Path, int]:
    """
    Downloads a shared file or folder from Google Drive and prepares it in input_root/<book_name>.
    Supports real-time status updates via status_callback.
    """
    def log(msg: str, meta: Optional[Dict[str, Any]] = None):
        if status_callback:
            status_callback(msg, meta)

    url = url.strip()
    input_root = Path(input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.gettempdir()) / f"gdrive_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    log("Đang phân tích đường dẫn Google Drive...", {"percent": 5, "status": "connecting"})

    try:
        is_folder = "folders/" in url
        if is_folder:
            folder_match = re.search(r"folders/([a-zA-Z0-9_-]+)", url)
            folder_id = folder_match.group(1) if folder_match else None
            if not folder_id:
                raise ValueError("Không tìm thấy Folder ID hợp lệ trong đường dẫn Google Drive.")

            log(f"Phát hiện thư mục Google Drive (ID: {folder_id[:8]}...). Đang kiểm tra...", {"percent": 8, "status": "connecting"})

            detected_folder_name = None
            try:
                sess, _ = _get_session(proxy=None, use_cookies=False, user_agent=MODERN_USER_AGENT, cookies_file=None)
                gf = _download_and_parse_google_drive_link(sess=sess, folder_id=folder_id, quiet=True, verify=True)
                if gf and gf.name:
                    detected_folder_name = gf.name.strip()
            except Exception:
                pass

            items = gdown.download_folder(
                url=url,
                skip_download=True,
                user_agent=MODERN_USER_AGENT,
                quiet=True
            )
            if not items:
                raise RuntimeError(
                    "Thư mục Google Drive trống hoặc chưa được mở quyền chia sẻ công khai: "
                    "'Bất kỳ ai có đường liên kết đều có thể xem'."
                )

            total_items = len(items)
            display_name = detected_folder_name or f"Thư mục ({folder_id[:8]})"
            log(f"Đã tìm thấy '{display_name}' ({total_items} tệp). Bắt đầu tải song song 4 luồng...", {
                "percent": 12,
                "status": "downloading",
                "total_str": f"{total_items} tệp"
            })

            downloaded_count = 0
            downloaded_bytes = 0
            start_time = time.time()
            progress_lock = threading.Lock()

            def _worker_task(item):
                nonlocal downloaded_count, downloaded_bytes
                dest = temp_dir / item.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                ok = download_gdrive_file(item.id, dest)
                with progress_lock:
                    if ok:
                        downloaded_count += 1
                        downloaded_bytes += dest.stat().st_size
                    elapsed = time.time() - start_time
                    spd = (downloaded_bytes / elapsed) if elapsed > 0 else 0.0
                    spd_str = f"{format_bytes(spd)}/s"
                    pct = round(12.0 + (downloaded_count / total_items) * 78.0, 1)
                    curr_str = format_bytes(downloaded_bytes)
                    msg = f"[{downloaded_count}/{total_items}] Tải '{item.path}' ({curr_str} • {spd_str})"
                    log(msg, {
                        "percent": pct,
                        "current_str": curr_str,
                        "total_str": f"{total_items} tệp",
                        "speed": spd_str,
                        "status": "downloading"
                    })

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(_worker_task, items))

            if downloaded_count == 0:
                raise RuntimeError("Không tải được tệp nào từ thư mục Google Drive. Vui lòng kiểm tra lại quyền truy cập.")

            log(f"Tải hoàn tất {downloaded_count}/{total_items} tệp ({format_bytes(downloaded_bytes)}). Đang quét định dạng...", {
                "percent": 90,
                "status": "extracting"
            })

            archive_exts = {".rar", ".zip", ".7z"}
            found_archives = [p for p in temp_dir.rglob("*") if p.is_file() and p.suffix.lower() in archive_exts]
            if found_archives:
                log(f"Tìm thấy file nén '{found_archives[0].name}'. Đang giải nén...", {"percent": 92, "status": "extracting"})
                return extract_archive(found_archives[0], input_root, custom_book_name=custom_book_name)

            image_files = [p for p in temp_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
            if not image_files:
                raise RuntimeError("Không tìm thấy file ảnh (.jpg, .png, .heic, v.v.) hay file nén nào trong thư mục Google Drive.")

            raw_book_name = custom_book_name.strip() if custom_book_name else (detected_folder_name or f"gdrive_book_{folder_id[:8]}")
            book_name = "".join(c for c in raw_book_name if c not in '<>:"/\\|?*').strip() or "GoogleDrive_Book"

            target_dir = input_root / book_name
            target_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(image_files, key=lambda p: natural_sort_key(p.name))
            log(f"Đang đồng bộ {len(image_files)} trang ảnh vào thư mục '{book_name}'...", {"percent": 95, "status": "extracting"})

            for idx, img_path in enumerate(image_files, 1):
                clean_filename = f"page_{idx:04d}{img_path.suffix.lower()}"
                dest_path = target_dir / clean_filename
                shutil.move(str(img_path), str(dest_path))

            log(f"Hoàn tất nạp dữ liệu cuốn '{book_name}'! Sẵn sàng OCR {len(image_files)} trang sách.", {
                "percent": 100,
                "status": "ready",
                "book_name": book_name,
                "images_count": len(image_files)
            })
            return book_name, target_dir, len(image_files)

        else:
            log("Đang kết nối Google Drive và kiểm tra tệp...", {"percent": 10, "status": "connecting"})
            file_id, is_gdrive = parse_url(url)
            output_pattern = str(temp_dir) + "/"

            tracker = DownloadProgressTracker(status_callback)
            downloaded_file = None

            if file_id:
                direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                sess = requests.Session()
                sess.headers.update({"User-Agent": MODERN_USER_AGENT})
                try:
                    head_res = sess.get(direct_url, stream=True, timeout=15)
                    if head_res.status_code == 200:
                        disp = head_res.headers.get("Content-Disposition", "")
                        fname_match = re.search(r'filename="?([^";]+)"?', disp)
                        fname = fname_match.group(1) if fname_match else f"download_{file_id}"
                        dest = temp_dir / fname
                        total_sz = int(head_res.headers.get("Content-Length", 0)) or None
                        cur_sz = 0
                        with open(dest, "wb") as f:
                            for chunk in head_res.iter_content(chunk_size=512 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    cur_sz += len(chunk)
                                    tracker(cur_sz, total_sz)
                        if dest.is_file() and dest.stat().st_size > 0:
                            downloaded_file = str(dest)
                except Exception:
                    pass

            if not downloaded_file:
                downloaded_file = gdown.download(
                    url=url,
                    output=output_pattern,
                    quiet=False,
                    use_cookies=False,
                    user_agent=MODERN_USER_AGENT,
                    progress=tracker
                )

            if not downloaded_file or not Path(downloaded_file).is_file():
                raise RuntimeError(
                    "Không thể tải file từ Google Drive. Vui lòng đảm bảo link được bật quyền: "
                    "'Bất kỳ ai có đường liên kết đều có thể xem'."
                )

            downloaded_path = Path(downloaded_file)
            ext = downloaded_path.suffix.lower()
            file_size_str = format_bytes(downloaded_path.stat().st_size)
            log(f"Tải file hoàn tất ({file_size_str})! Đang kiểm tra định dạng...", {"percent": 92, "status": "extracting"})

            if ext in {".rar", ".zip", ".7z"}:
                log(f"Đang giải nén file nén '{downloaded_path.name}'...", {"percent": 95, "status": "extracting"})
                b_name, t_dir, count = extract_archive(downloaded_path, input_root, custom_book_name=custom_book_name)
                log(f"Giải nén thành công {count} trang ảnh từ file '{downloaded_path.name}'.", {"percent": 100, "status": "ready"})
                return b_name, t_dir, count
            elif ext in VALID_IMAGE_EXTS:
                raw_name = custom_book_name.strip() if custom_book_name else downloaded_path.stem
                raw_name = re.sub(r"-\d{8}T\d{6}Z.*$", "", raw_name)
                book_name = "".join(c for c in raw_name if c not in '<>:"/\\|?*').strip() or "GoogleDrive_Book"
                target_dir = input_root / book_name
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / downloaded_path.name
                shutil.move(str(downloaded_path), str(dest))
                log(f"Đã lưu ảnh trang sách vào '{book_name}'.", {"percent": 100, "status": "ready"})
                return book_name, target_dir, 1
            else:
                try:
                    log(f"Đang thử giải nén file '{downloaded_path.name}'...", {"percent": 95, "status": "extracting"})
                    b_name, t_dir, count = extract_archive(downloaded_path, input_root, custom_book_name=custom_book_name)
                    log(f"Giải nén thành công {count} trang ảnh.", {"percent": 100, "status": "ready"})
                    return b_name, t_dir, count
                except Exception:
                    raise RuntimeError(
                        f"File tải về '{downloaded_path.name}' không phải là file nén (.rar, .zip, .7z) hay file ảnh hợp lệ."
                    )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
