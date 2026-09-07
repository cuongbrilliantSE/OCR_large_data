import threading
import time
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config import AppConfig
from src.db.state_tracker import StateTracker
from src.utils.file_utils import scan_image_files, natural_sort_key
from src.pipeline.dispatcher import Dispatcher

class OCRJobManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OCRJobManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: str = "configs/config.yaml"):
        if getattr(self, "_initialized", False):
            return
        self.config_path = config_path
        self.config = AppConfig.load_from_yaml(config_path)
        self.tracker = StateTracker(self.config.paths.db_path)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.job_threads: Dict[str, threading.Thread] = {}
        self._initialized = True

    def get_or_create_job(self, book_name: str) -> Dict[str, Any]:
        if book_name not in self.active_jobs:
            self.active_jobs[book_name] = {
                "book_name": book_name,
                "state": "idle",  # idle, processing, paused, completed, error
                "total": 0,
                "done": 0,
                "pending": 0,
                "failed": 0,
                "percent": 0.0,
                "avg_time_ms": 0.0,
                "latest_preview": None,
                "logs": [],
                "error": None,
                "output_text_file": None,
                "output_zip_file": None,
                "updated_at": time.time()
            }
        return self.active_jobs[book_name]

    def add_log(self, book_name: str, message: str):
        job = self.get_or_create_job(book_name)
        timestamp = time.strftime("%H:%M:%S")
        job["logs"].append(f"[{timestamp}] {message}")
        if len(job["logs"]) > 100:
            job["logs"] = job["logs"][-100:]
        job["updated_at"] = time.time()

    def scan_and_register(self, book_name: str) -> int:
        scanned_items = list(scan_image_files(self.config.paths.input_dir, book_name=book_name))
        count = self.tracker.register_files(scanned_items)
        self.refresh_stats(book_name)
        return len(scanned_items)

    def refresh_stats(self, book_name: str) -> Dict[str, Any]:
        job = self.get_or_create_job(book_name)
        stats = self.tracker.get_statistics(book_name=book_name)
        job["total"] = stats["total"]
        job["done"] = stats["done"]
        job["pending"] = stats["pending"]
        job["failed"] = stats["failed"]
        job["percent"] = stats["percent_complete"]
        job["avg_time_ms"] = stats["avg_time_ms"]
        job["updated_at"] = time.time()

        # Check latest generated text file for preview
        output_book_dir = Path(self.config.paths.output_dir) / book_name
        if output_book_dir.is_dir():
            txt_files = list(output_book_dir.glob("*.txt"))
            if txt_files:
                latest_txt = max(txt_files, key=lambda p: p.stat().st_mtime)
                try:
                    snippet = latest_txt.read_text(encoding="utf-8", errors="replace")[:600]
                    job["latest_preview"] = {
                        "filename": latest_txt.name,
                        "snippet": snippet
                    }
                except Exception:
                    pass

        merged_txt = Path(self.config.paths.output_dir) / f"{book_name}.txt"
        if merged_txt.is_file():
            job["output_text_file"] = str(merged_txt)

        pkg_zip = Path(self.config.paths.output_dir) / f"{book_name}_package.zip"
        if pkg_zip.is_file():
            job["output_zip_file"] = str(pkg_zip)

        if job["total"] > 0 and job["done"] == job["total"]:
            job["state"] = "completed"
        elif job["failed"] > 0 and job["pending"] == 0 and job["state"] != "processing":
            job["state"] = "has_failed"

        return job

    def start_job(self, book_name: str, num_workers: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            job = self.get_or_create_job(book_name)
            if job["state"] == "processing":
                return job

            self.scan_and_register(book_name)
            self.stop_events[book_name] = threading.Event()
            job["state"] = "processing"
            job["error"] = None
            self.add_log(book_name, f"Khởi chạy tiến trình OCR cho cuốn sách '{book_name}'...")

            t = threading.Thread(
                target=self._run_job_thread,
                args=(book_name, num_workers or self.config.system.num_workers),
                daemon=True
            )
            self.job_threads[book_name] = t
            t.start()
            return job

    def retry_failed_job(self, book_name: str, num_workers: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            count = self.tracker.retry_failed_tasks(book_name=book_name)
            self.add_log(book_name, f"Đã đặt lại {count} trang lỗi về trạng thái chờ xử lý (PENDING).")
            return self.start_job(book_name, num_workers=num_workers)

    def pause_job(self, book_name: str) -> Dict[str, Any]:
        with self._lock:
            job = self.get_or_create_job(book_name)
            if book_name in self.stop_events:
                self.stop_events[book_name].set()
            job["state"] = "paused"
            self.add_log(book_name, "Tiến trình đã được yêu cầu tạm dừng.")
            return job

    def _run_job_thread(self, book_name: str, num_workers: int):
        cfg = AppConfig.load_from_yaml(self.config_path)
        cfg.system.num_workers = num_workers

        stop_event = self.stop_events.get(book_name)

        try:
            # Dispatch batches while tracking progress
            while not (stop_event and stop_event.is_set()):
                self.refresh_stats(book_name)
                job = self.active_jobs[book_name]

                # Check if done
                if job["pending"] == 0 and job["failed"] == 0 and job["total"] > 0:
                    break

                # Fetch next pending tasks batch
                pending_tasks = self.tracker.get_pending_tasks(limit=cfg.system.batch_size, book_name=book_name)
                if not pending_tasks:
                    break

                # Run dispatcher for these tasks
                dispatcher = Dispatcher(cfg)
                dispatcher.run(max_limit=cfg.system.batch_size, book_name=book_name)
                self.refresh_stats(book_name)

                # Small cooldown
                time.sleep(0.5)

            # Finished or stopped
            self.refresh_stats(book_name)
            job = self.active_jobs[book_name]

            if stop_event and stop_event.is_set():
                job["state"] = "paused"
                self.add_log(book_name, "Tiến trình đã tạm dừng an toàn. Các trang đã xong được giữ nguyên.")
            elif job["total"] > 0 and job["done"] == job["total"]:
                job["state"] = "completed"
                self.add_log(book_name, "Toàn bộ các trang sách đã được OCR hoàn tất 100%!")
                # Automatically package book
                self.merge_and_package_book(book_name)
            elif job["failed"] > 0 and job["pending"] == 0:
                job["state"] = "has_failed"
                self.add_log(book_name, f"Đã dừng. Có {job['failed']} trang bị lỗi cần thử lại.")
            else:
                job["state"] = "idle"

        except Exception as e:
            import traceback
            traceback.print_exc()
            job = self.get_or_create_job(book_name)
            job["state"] = "error"
            job["error"] = str(e)
            self.add_log(book_name, f"Lỗi trong quá trình xử lý: {str(e)}")


    def merge_and_package_book(self, book_name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Merges all individual page txt files into <book_name>.txt and packages into <book_name>_package.zip.
        """
        output_dir = Path(self.config.paths.output_dir)
        book_output_dir = output_dir / book_name

        if not book_output_dir.is_dir():
            return None, None

        txt_files = [f for f in book_output_dir.rglob("*.txt") if f.is_file()]
        if not txt_files:
            return None, None

        txt_files = sorted(txt_files, key=lambda p: natural_sort_key(p.name))

        # 1. Merge into single file
        merged_txt_path = output_dir / f"{book_name}.txt"
        with open(merged_txt_path, "w", encoding="utf-8") as out:
            for idx, f in enumerate(txt_files, 1):
                content = f.read_text(encoding="utf-8", errors="replace").strip()
                out.write(f"=== [Trang {idx}/{len(txt_files)}] {f.name} ===\n\n")
                out.write(content)
                out.write("\n\n")

        # 2. Package into ZIP
        zip_path = output_dir / f"{book_name}_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add merged text file
            zf.write(merged_txt_path, arcname=f"{book_name}.txt")
            # Add individual pages
            for f in txt_files:
                zf.write(f, arcname=f"pages/{f.name}")

        job = self.get_or_create_job(book_name)
        job["output_text_file"] = str(merged_txt_path)
        job["output_zip_file"] = str(zip_path)
        self.add_log(book_name, f"Đã đóng gói hoàn chỉnh file sách ({len(txt_files)} trang)!")
        return merged_txt_path, zip_path

    def list_books(self) -> List[Dict[str, Any]]:
        # Sync database and input folders
        scanned_items = list(scan_image_files(self.config.paths.input_dir))
        if scanned_items:
            self.tracker.register_files(scanned_items)

        stats = self.tracker.get_book_statistics()
        output_dir = Path(self.config.paths.output_dir)

        result = []
        for s in stats:
            b_name = s["book_name"]
            merged_file = output_dir / f"{b_name}.txt"
            zip_file = output_dir / f"{b_name}_package.zip"
            job = self.active_jobs.get(b_name, {})
            
            if job.get("state") == "processing":
                status = "processing"
            elif s["done"] == s["total"] and s["total"] > 0:
                status = "completed"
            elif s["failed"] > 0 and s["pending"] == 0:
                status = "has_failed"
            elif job.get("state") == "paused":
                status = "paused"
            elif s["done"] > 0:
                status = "paused"
            else:
                status = "idle"
            
            result.append({
                "book_name": b_name,
                "total": s["total"],
                "done": s["done"],
                "pending": s["pending"],
                "failed": s["failed"],
                "percent": s["percent"],
                "status": status,
                "has_text_file": merged_file.is_file(),
                "has_zip_file": zip_file.is_file()
            })
        return result

