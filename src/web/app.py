import os
import re
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.web.archive_extractor import extract_archive
from src.web.gdrive_downloader import is_gdrive_url, download_and_extract_gdrive, gdrive_task_manager
from src.web.job_manager import OCRJobManager
from src.db.book_repository import BookRepository, slugify
from src.db.hf_dataset_sync import HFDatasetSync

app = FastAPI(
    title="Book OCR Studio API",
    description="High-throughput Large-Scale Image to Text OCR Pipeline",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Job Manager & Book Repository
manager = OCRJobManager()
book_repo = BookRepository()
dataset_syncer = HFDatasetSync()

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
LIBRARY_HTML = STATIC_DIR / "library.html"
READER_HTML = STATIC_DIR / "reader.html"
MUSEUM_HTML = STATIC_DIR / "museum.html"
EXHIBIT_HTML = STATIC_DIR / "exhibit.html"

# Mount static covers directory
COVERS_DIR = Path("data/covers")
COVERS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/covers", StaticFiles(directory=str(COVERS_DIR)), name="covers")


def extract_gdrive_id(url: str) -> Optional[str]:
    """Extract folder ID or file ID from Google Drive URL."""
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


@app.on_event("startup")
async def on_startup():
    """Sync local books, pull cloud manifest, and seed museum metadata on startup."""
    try:
        book_repo.sync_all_books()
        if dataset_syncer.is_configured():
            dataset_syncer.pull_manifest_and_restore(book_repo)
        book_repo.seed_museum_metadata()
    except Exception as e:
        print(f"[App Startup] Sync error: {e}")


# ── Web Pages ────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_index():
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.api_route("/library", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_library():
    if not LIBRARY_HTML.is_file():
        raise HTTPException(status_code=404, detail="Library HTML not found")
    return HTMLResponse(content=LIBRARY_HTML.read_text(encoding="utf-8"))


@app.api_route("/books/{slug}", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_reader(slug: str):
    if not READER_HTML.is_file():
        raise HTTPException(status_code=404, detail="Reader HTML not found")

    book = book_repo.get_book_by_slug(slug)
    if not book and dataset_syncer.is_configured():
        dataset_syncer.pull_manifest_and_restore(book_repo)
        book = book_repo.get_book_by_slug(slug)

    if not book:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy cuốn sách với mã: {slug}")

    html = READER_HTML.read_text(encoding="utf-8")
    title = book.get("title", "")
    category = book.get("category", "Tài liệu")
    summary = book.get("summary", "") or "Cuốn sách đã được số hóa hoàn chỉnh bằng AI."
    cover = book.get("cover_image") or "/covers/default-book.jpg"
    total_pages = str(book.get("total_pages", 0))
    word_count = f"{book.get('word_count', 0):,}"

    html = html.replace("{{BOOK_TITLE}}", title)
    html = html.replace("{{BOOK_CATEGORY}}", category)
    html = html.replace("{{BOOK_SUMMARY}}", summary)
    html = html.replace("{{BOOK_COVER}}", cover)
    html = html.replace("{{TOTAL_PAGES}}", total_pages)
    html = html.replace("{{WORD_COUNT}}", word_count)
    html = html.replace("{{BOOK_SLUG}}", slug)
    html = html.replace("{{BOOK_TITLE_URL}}", urllib.parse.quote(title))

    return HTMLResponse(content=html)


@app.api_route("/museum", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_museum():
    if not MUSEUM_HTML.is_file():
        raise HTTPException(status_code=404, detail="Giao diện Bảo Tàng Số không tồn tại")
    return HTMLResponse(content=MUSEUM_HTML.read_text(encoding="utf-8"))


@app.api_route("/museum/exhibit/{slug}", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def get_museum_exhibit(slug: str):
    if not EXHIBIT_HTML.is_file():
        raise HTTPException(status_code=404, detail="Giao diện Phòng Triển Lãm không tồn tại")

    book = book_repo.get_book_by_slug(slug)
    if not book and dataset_syncer.is_configured():
        dataset_syncer.pull_manifest_and_restore(book_repo)
        book = book_repo.get_book_by_slug(slug)

    if not book:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy hiện vật di sản: {slug}")

    # Ensure book text file is ready locally / on cloud
    if dataset_syncer.is_configured():
        dataset_syncer.ensure_book_text_file(slug, book["title"])

    html = EXHIBIT_HTML.read_text(encoding="utf-8")
    title = book.get("title", "")
    artifact_code = book.get("artifact_code") or f"BT-{slug[:6].upper()}"
    period_era = book.get("period_era") or "Chưa rõ niên đại"

    html = html.replace("{{BOOK_TITLE}}", title)
    html = html.replace("{{ARTIFACT_CODE}}", artifact_code)
    html = html.replace("{{PERIOD_ERA}}", period_era)
    html = html.replace("{{BOOK_SLUG}}", slug)

    return HTMLResponse(content=html)


# ── Library API Endpoints ────────────────────────────────────────────────────

@app.get("/api/library/books")
async def api_library_books(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: str = Query("newest"),
    limit: int = Query(50),
    offset: int = Query(0)
):
    return JSONResponse(book_repo.list_books(category=category, search=search, sort=sort, limit=limit, offset=offset))


@app.get("/api/library/categories")
async def api_library_categories():
    return JSONResponse(book_repo.get_categories_stats())


@app.get("/api/library/books/{slug}")
async def api_library_book_detail(slug: str):
    book = book_repo.get_book_by_slug(slug)
    if not book:
        raise HTTPException(status_code=404, detail="Không tìm thấy sách")
    return JSONResponse(book)


@app.get("/api/library/books/{slug}/text")
async def api_library_book_text(slug: str):
    book = book_repo.get_book_by_slug(slug)
    if not book:
        raise HTTPException(status_code=404, detail="Không tìm thấy sách")

    title = book["title"]
    # Ensure text exists locally or pull from HF Dataset
    text_path = dataset_syncer.ensure_book_text_file(slug, title)
    if not text_path or not text_path.is_file():
        text_path = Path(manager.config.paths.output_dir) / f"{title}.txt"

    if not text_path.is_file():
        # Attempt to package if pages exist
        text_path, _ = manager.merge_and_package_book(title)

    if not text_path or not text_path.is_file():
        raise HTTPException(status_code=404, detail="Nội dung văn bản sách chưa sẵn sàng")

    return PlainTextResponse(text_path.read_text(encoding="utf-8", errors="ignore"))


@app.patch("/api/library/books/{slug}")
async def api_library_update_book(
    slug: str,
    payload: dict = Body(...)
):
    updated = book_repo.update_metadata(
        slug=slug,
        title=payload.get("title"),
        category=payload.get("category"),
        tags=payload.get("tags"),
        summary=payload.get("summary"),
        is_public=payload.get("is_public")
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Không tìm thấy sách để cập nhật")

    # Sync updated metadata to cloud
    if dataset_syncer.is_configured():
        dataset_syncer.push_book_async(slug)

    return JSONResponse(updated)


# ── Museum API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/museum/exhibits")
async def api_museum_exhibits():
    """Retrieve thematic exhibition halls and curated artifacts."""
    return JSONResponse(book_repo.get_museum_exhibits())


@app.get("/api/museum/artifacts/{slug}")
async def api_museum_artifact_dossier(slug: str):
    """Retrieve full artifact dossier with 1:1 original page scans and OCR text."""
    book = book_repo.get_book_by_slug(slug)
    if not book and dataset_syncer.is_configured():
        dataset_syncer.pull_manifest_and_restore(book_repo)
        book = book_repo.get_book_by_slug(slug)

    if not book:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy hồ sơ hiện vật '{slug}'")

    if dataset_syncer.is_configured():
        dataset_syncer.ensure_book_text_file(slug, book["title"])

    dossier = book_repo.get_artifact_dossier(slug)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy hồ sơ hiện vật '{slug}'")
    return JSONResponse(dossier)


@app.get("/api/museum/artifacts/{slug}/page-image/{page_num}")
async def api_museum_page_image(slug: str, page_num: int):
    """Serve high-resolution original page scan (auto-converted to JPEG)."""
    if page_num == 1 and dataset_syncer.is_configured():
        dataset_syncer.ensure_book_cover(slug)

    img_path = book_repo.get_page_image_path(slug, page_num)
    if not img_path or not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh gốc cho trang {page_num}")
    return FileResponse(
        path=img_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.post("/api/museum/artifacts/{slug}/chat")
async def api_museum_chat(
    slug: str,
    payload: dict = Body(...)
):
    """AI Museum Curator interactive Q&A grounded on artifact content."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Thiếu câu hỏi của khách tham quan.")

    dossier = book_repo.get_artifact_dossier(slug)
    if not dossier:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy hiện vật '{slug}'")

    meta = dossier.get("artifact", {})
    title = meta.get("title", "")
    code = meta.get("artifact_code", "")
    era = meta.get("period_era", "")
    provenance = meta.get("provenance", "")
    curator_notes = meta.get("curator_notes", "")
    highlights = meta.get("highlights_json") or []

    # Read partial book text for grounding
    sample_text = ""
    pages = dossier.get("pages", [])
    if pages:
        sample_pages = [p.get("text", "") for p in pages[:12] if p.get("text")]
        sample_text = "\n\n--- Trích đoạn nội dung hiện vật ---\n\n" + "\n\n".join(sample_pages)[:6000]

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"""Bạn là Giám tuyển Di sản (AI Museum Curator) của Bảo tàng Di sản số, đang thuyết minh và đối thoại chuyên sâu cùng khách tham quan về hiện vật quý:
Tên hiện vật: {title}
Mã lưu trữ: {code}
Niên đại: {era}
Nguồn gốc / Xuất xứ: {provenance}
Ghi chú giám tuyển: {curator_notes}
Điểm nhấn: {", ".join(highlights) if isinstance(highlights, list) else ""}

{sample_text}

Khách tham quan hỏi: "{question}"

Yêu cầu trả lời:
- Giọng văn trang trọng, uyên bác, trang nhã, đúng phong thái giám tuyển bảo tàng văn hóa - lịch sử Việt Nam.
- Dẫn chứng chuẩn xác từ ghi chú giám tuyển và trích đoạn hiện vật.
- Trả lời ngắn gọn, khúc chiết, súc tích (khoảng 2-4 đoạn văn).
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = (response.text or "").strip()
            if answer:
                return JSONResponse({"answer": answer})
        except Exception as e:
            print(f"[Museum Chat Error] {e}")

    fallback_answer = (
        f"Kính chào quý khách. Về hiện vật **{title}** ({code}, niên đại {era}), "
        f"ghi chép giám tuyển ghi nhận: {curator_notes}. "
        f"Hiện vật có nguồn gốc từ {provenance}. "
        f"Quý khách có thể sử dụng chế độ 'So sánh đối chiếu' (Dual-View) phóng đại trang sách để trực tiếp quan sát văn bản nguyên tác."
    )
    return JSONResponse({"answer": fallback_answer})


# ── OCR Pipeline API Endpoints ───────────────────────────────────────────────

@app.post("/api/upload")
async def upload_archive(
    file: UploadFile = File(...),
    book_name: Optional[str] = Form(None),
    workers: int = Form(2),
    force: bool = Form(False)
):
    original_filename = file.filename
    ext = Path(original_filename).suffix.lower()

    if ext not in (".rar", ".zip", ".7z", ".tar", ".gz"):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không được hỗ trợ: {ext}. Vui lòng tải lên file .rar, .zip hoặc .7z."
        )

    # Deduplication check by book name
    target_name = (book_name or Path(original_filename).stem).strip()
    if not force:
        dup = book_repo.find_duplicate(title=target_name)
        if dup and dup.get("status") == "COMPLETED":
            return JSONResponse({
                "status": "already_exists",
                "message": f"Cuốn sách '{dup['title']}' đã có sẵn trong Thư viện!",
                "book": dup,
                "reader_url": f"/books/{dup['slug']}"
            })

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

        if not force:
            dup = book_repo.find_duplicate(title=actual_book_name)
            if dup and dup.get("status") == "COMPLETED":
                return JSONResponse({
                    "status": "already_exists",
                    "message": f"Cuốn sách '{dup['title']}' đã có sẵn trong Thư viện!",
                    "book": dup,
                    "reader_url": f"/books/{dup['slug']}"
                })

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
    workers: int = Form(2),
    force: bool = Form(False)
):
    clean_url = url.strip()
    if not clean_url or not is_gdrive_url(clean_url):
        raise HTTPException(
            status_code=400,
            detail="Đường dẫn không hợp lệ. Vui lòng nhập link chia sẻ Google Drive (dạng file hoặc folder)."
        )

    # Deduplication Check
    gdrive_id = extract_gdrive_id(clean_url)
    if not force:
        dup = book_repo.find_duplicate(source_id=gdrive_id, title=book_name)
        if dup and dup.get("status") == "COMPLETED":
            return JSONResponse({
                "status": "already_exists",
                "message": f"Cuốn sách '{dup['title']}' đã có sẵn trong Thư viện!",
                "book": dup,
                "reader_url": f"/books/{dup['slug']}"
            })

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

            # Store source_id mapping in book_repo
            book_repo.upsert_book(
                title=actual_book_name,
                total_pages=image_count,
                source_type="gdrive",
                source_id=gdrive_id or "",
                status="PROCESSING"
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
                clean_err = "Không thể tải từ Google Drive do quyền riêng tư. Vui lòng chuyển link sang chế độ: 'Bất kỳ ai có đường liên kết đều có thể xem' (Anyone with the link can view)."
            else:
                clean_err = f"Lỗi tải từ Google Drive: {err_msg}"

            gdrive_task_manager.update_task(task_id, clean_err, {
                "status": "error",
                "error": clean_err
            })

    import threading
    threading.Thread(target=_worker, daemon=True).start()

    return JSONResponse({
        "status": "accepted",
        "task_id": task_id,
        "message": "Đang kết nối và tải ảnh từ Google Drive..."
    })


@app.get("/api/gdrive/status/{task_id}")
async def get_gdrive_status(task_id: str):
    task = gdrive_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ tải Drive.")
    return JSONResponse(task)


@app.get("/api/jobs/{book_name}")
async def get_job_status(book_name: str):
    job = manager.get_job(book_name)
    if not job:
        job = manager.get_or_create_job(book_name)
        manager.refresh_stats(book_name)
    return JSONResponse(job)


@app.post("/api/pause/{book_name}")
async def pause_job(book_name: str):
    job = manager.pause_job(book_name)
    return JSONResponse(job)


@app.post("/api/resume/{book_name}")
async def resume_job(book_name: str, workers: int = Form(2)):
    job = manager.start_job(book_name, num_workers=workers)
    return JSONResponse(job)


@app.post("/api/retry/{book_name}")
async def retry_job(book_name: str, workers: int = Form(2)):
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
    txt_path = Path(manager.config.paths.output_dir) / f"{book_name}.txt"
    if not txt_path.is_file():
        txt_path, _ = manager.merge_and_package_book(book_name)

    if not txt_path or not txt_path.is_file():
        slug = slugify(book_name)
        txt_path = dataset_syncer.ensure_book_text_file(slug, book_name)

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
        _, zip_path = manager.merge_and_package_book(book_name)

    if not zip_path or not zip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file zip của sách '{book_name}'.")

    return FileResponse(
        path=zip_path,
        filename=f"{book_name}_package.zip",
        media_type="application/zip"
    )
