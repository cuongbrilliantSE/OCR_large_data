import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.web.archive_extractor import extract_archive
from src.web.gdrive_downloader import is_gdrive_url, download_and_extract_gdrive, gdrive_task_manager
from src.web.job_manager import OCRJobManager



app = FastAPI(
    title="Book OCR Studio API",
    description="High-throughput Large-Scale Image to Text OCR Pipeline with Gemini Vision AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Job Manager
manager = OCRJobManager()

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@app.get("/", response_class=HTMLResponse)
async def get_index():
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_archive(
    file: UploadFile = File(...),
    book_name: Optional[str] = Form(None),
    workers: int = Form(2)
):
    """
    Receives an uploaded archive (.rar, .zip, .7z), extracts it, and starts the OCR pipeline.
    """
    original_filename = file.filename
    ext = Path(original_filename).suffix.lower()

    if ext not in (".rar", ".zip", ".7z", ".tar", ".gz"):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không được hỗ trợ: {ext}. Vui lòng tải lên file .rar, .zip hoặc .7z."
        )

    # Save uploaded file to temp file
    temp_dir = Path(tempfile.gettempdir())
    temp_archive_path = temp_dir / f"upload_{os.getpid()}_{original_filename}"

    try:
        with open(temp_archive_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        input_root = Path(manager.config.paths.input_dir)
        actual_book_name, target_dir, image_count = extract_archive(
            archive_path=temp_archive_path,
            input_root=input_root,
            custom_book_name=book_name
        )

        # Start OCR job
        manager.add_log(actual_book_name, f"Đã giải nén thành công {image_count} file ảnh vào thư mục '{actual_book_name}'.")
        job = manager.start_job(actual_book_name, num_workers=workers)

        return JSONResponse({
            "status": "success",
            "book_name": actual_book_name,
            "images_count": image_count,
            "job": job
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_archive_path.exists():
            temp_archive_path.unlink(missing_ok=True)


@app.post("/api/gdrive/start")
async def start_gdrive_download(
    url: str = Form(...),
    book_name: Optional[str] = Form(None),
    workers: int = Form(2)
):
    """
    Initiates asynchronous Google Drive download and extraction, returning task_id for real-time tracking.
    """
    clean_url = url.strip()
    if not clean_url or not is_gdrive_url(clean_url):
        raise HTTPException(
            status_code=400,
            detail="Đường dẫn không hợp lệ. Vui lòng nhập link chia sẻ Google Drive (dạng file hoặc folder)."
        )

    task_id = gdrive_task_manager.create_task()

    def _worker():
        def cb(msg: str, meta: Optional[dict] = None):
            gdrive_task_manager.update_task(task_id, msg, meta)

        try:
            input_root = Path(manager.config.paths.input_dir)
            actual_book_name, target_dir, image_count = download_and_extract_gdrive(
                url=clean_url,
                input_root=input_root,
                custom_book_name=book_name,
                status_callback=cb
            )

            gdrive_task_manager.update_task(task_id, f"Hoàn tất tải về và giải nén {image_count} trang ảnh! Đang khởi chạy OCR...", {
                "status": "completed",
                "percent": 100.0,
                "book_name": actual_book_name,
                "images_count": image_count
            })

            manager.add_log(actual_book_name, f"Đã nạp thành công {image_count} trang ảnh từ Google Drive.")
            manager.start_job(actual_book_name, num_workers=workers)

        except Exception as e:
            err_msg = str(e)
            if any(keyword in err_msg for keyword in ["Cannot retrieve the public link", "Permission denied", "Failed to retrieve", "FileURLRetrievalError"]):
                err_msg = "Không thể truy cập file/thư mục trên Google Drive. Vui lòng kiểm tra lại link và đảm bảo quyền chia sẻ đã được bật: 'Bất kỳ ai có đường liên kết đều có thể xem' (Anyone with the link can view)."
            gdrive_task_manager.update_task(task_id, f"Lỗi: {err_msg}", {
                "status": "error",
                "error": err_msg
            })

    import threading
    threading.Thread(target=_worker, daemon=True).start()

    return JSONResponse({
        "status": "started",
        "task_id": task_id
    })


@app.get("/api/gdrive/status/{task_id}")
async def get_gdrive_status(task_id: str):
    task = gdrive_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ tải Google Drive này.")
    return JSONResponse(task)


@app.post("/api/gdrive")
async def process_gdrive_link(
    url: str = Form(...),
    book_name: Optional[str] = Form(None),
    workers: int = Form(2)
):
    """
    Synchronous fallback for Google Drive download and extraction.
    """
    clean_url = url.strip()
    if not clean_url or not is_gdrive_url(clean_url):
        raise HTTPException(
            status_code=400,
            detail="Đường dẫn không hợp lệ. Vui lòng nhập link chia sẻ Google Drive (dạng file hoặc folder)."
        )

    try:
        input_root = Path(manager.config.paths.input_dir)
        actual_book_name, target_dir, image_count = download_and_extract_gdrive(
            url=clean_url,
            input_root=input_root,
            custom_book_name=book_name
        )

        manager.add_log(actual_book_name, f"Đã kéo thành công từ Google Drive {image_count} file ảnh vào thư mục '{actual_book_name}'.")
        job = manager.start_job(actual_book_name, num_workers=workers)

        return JSONResponse({
            "status": "success",
            "book_name": actual_book_name,
            "images_count": image_count,
            "job": job
        })

    except Exception as e:
        err_msg = str(e)
        if any(keyword in err_msg for keyword in ["Cannot retrieve the public link", "Permission denied", "Failed to retrieve", "FileURLRetrievalError"]):
            err_msg = "Không thể truy cập file/thư mục trên Google Drive. Vui lòng kiểm tra lại link và đảm bảo quyền chia sẻ đã được bật: 'Bất kỳ ai có đường liên kết đều có thể xem' (Anyone with the link can view)."
        raise HTTPException(status_code=400, detail=err_msg)




@app.get("/api/status/{book_name}")

async def get_status(book_name: str):
    job = manager.refresh_stats(book_name)
    return JSONResponse(job)


@app.post("/api/start/{book_name}")
async def start_ocr(book_name: str, workers: Optional[int] = Form(2)):
    job = manager.start_job(book_name, num_workers=workers)
    return JSONResponse(job)


@app.post("/api/pause/{book_name}")
async def pause_ocr(book_name: str):
    job = manager.pause_job(book_name)
    return JSONResponse(job)


@app.post("/api/resume/{book_name}")
async def resume_ocr(book_name: str, workers: Optional[int] = Form(2)):
    job = manager.start_job(book_name, num_workers=workers)
    return JSONResponse(job)


@app.post("/api/retry/{book_name}")
async def retry_failed_ocr(book_name: str, workers: Optional[int] = Form(2)):
    job = manager.retry_failed_job(book_name, num_workers=workers)
    return JSONResponse(job)


@app.post("/api/merge/{book_name}")
async def merge_book(book_name: str):
    txt_path, zip_path = manager.merge_and_package_book(book_name)
    if not txt_path:
        raise HTTPException(status_code=400, detail=f"Chưa có trang văn bản nào hoàn thành cho cuốn sách '{book_name}'.")
    return JSONResponse({
        "status": "merged",
        "text_file": str(txt_path),
        "zip_file": str(zip_path)
    })


@app.get("/api/books")
async def list_books():
    books = manager.list_books()
    return JSONResponse(books)


@app.get("/api/download/text/{book_name}")
async def download_text(book_name: str):
    # Ensure merged text exists
    txt_path = Path(manager.config.paths.output_dir) / f"{book_name}.txt"
    if not txt_path.is_file():
        # Try merging on demand
        txt_path, _ = manager.merge_and_package_book(book_name)

    if not txt_path or not txt_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file text hoàn chỉnh của sách '{book_name}'.")

    return FileResponse(
        path=txt_path,
        filename=f"{book_name}.txt",
        media_type="text/plain; charset=utf-8"
    )


@app.get("/api/download/zip/{book_name}")
async def download_zip(book_name: str):
    zip_path = Path(manager.config.paths.output_dir) / f"{book_name}_package.zip"
    if not zip_path.is_file():
        # Package on demand
        _, zip_path = manager.merge_and_package_book(book_name)

    if not zip_path or not zip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file zip của sách '{book_name}'.")

    return FileResponse(
        path=zip_path,
        filename=f"{book_name}_package.zip",
        media_type="application/zip"
    )
