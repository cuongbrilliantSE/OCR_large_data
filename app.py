import os
import re
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

try:
    import spaces

    @spaces.GPU(duration=1)
    def _noop():
        """ZeroGPU requires at least one decorated function on Hugging Face."""
        pass
except ImportError:
    pass

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Initialize HEIC opener if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.engines.gemini_engine import GeminiVisionEngine
from src.preprocessor.image_cleaner import ImageCleaner
from src.web.gdrive_downloader import is_gdrive_url, download_and_extract_gdrive
from src.utils.file_utils import natural_sort_key


def ocr_single_image(
    image_file,
    split_double_pages: bool = False,
    auto_orient: bool = True,
    custom_api_key: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """
    Perform OCR on an uploaded image using Gemini Vision AI.
    Returns:
      (extracted_text, info_summary, output_txt_filepath)
    """
    if image_file is None:
        return "⚠️ Vui lòng tải lên một file ảnh sách.", "Chưa có ảnh.", None

    api_key = (custom_api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return (
            "⚠️ Chưa có Gemini API Key!\n\n"
            "Vui lòng nhập API Key của bạn vào ô 'Gemini API Key (Tùy chọn)' ở góc trái, "
            "hoặc cấu hình GEMINI_API_KEY trong Space Secrets tại huggingface.co.\n"
            "Bạn có thể lấy key miễn phí tại: https://aistudio.google.com/app/apikey",
            "Thiếu API Key",
            None
        )

    t0 = time.time()
    try:
        engine = GeminiVisionEngine(api_key=api_key)
        cleaner = ImageCleaner(
            max_dimension=2560,
            auto_orient=auto_orient,
            split_double_pages=split_double_pages
        )

        img_path = Path(image_file)
        img_bgr, meta = cleaner.load_and_preprocess(img_path)

        # Check if double page split requested and detected
        if split_double_pages and meta.get("is_double_page", False):
            pages = cleaner.split_pages(img_bgr)
            results = []
            for idx, p in enumerate(pages, 1):
                label = "TRANG TRÁI" if idx == 1 else "TRANG PHẢI"
                res = engine.recognize(p)
                results.append(f"=== [{label}] ===\n\n{res['full_text'].strip()}")
            full_text = "\n\n".join(results)
        else:
            res = engine.recognize(img_bgr)
            full_text = res["full_text"].strip()

        elapsed = round(time.time() - t0, 2)
        info = (
            f"⚡ Thời gian xử lý: {elapsed} giây | "
            f"Kích thước: {meta.get('processed_width', 0)}x{meta.get('processed_height', 0)} px | "
            f"Tự động xoay: {'Có' if meta.get('rotated_sideways') else 'Không'}"
        )

        # Save to temporary txt file for download
        tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp_txt.write(full_text)
        tmp_txt.close()

        return full_text, info, tmp_txt.name

    except Exception as e:
        return f"❌ Lỗi trong quá trình xử lý: {str(e)}", f"Lỗi: {str(e)}", None


def ocr_multiple_images(
    image_files: List,
    split_double_pages: bool = False,
    custom_api_key: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    Perform batch OCR on multiple uploaded book pages.
    """
    if not image_files:
        return "⚠️ Vui lòng tải lên ít nhất một file ảnh.", "Chưa có ảnh.", None

    api_key = (custom_api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return (
            "⚠️ Chưa có Gemini API Key. Vui lòng nhập API Key vào ô cấu hình.",
            "Thiếu API Key",
            None
        )

    t0 = time.time()
    engine = GeminiVisionEngine(api_key=api_key)
    cleaner = ImageCleaner(max_dimension=2560, auto_orient=True, split_double_pages=split_double_pages)

    sorted_files = sorted([Path(f) for f in image_files], key=lambda p: natural_sort_key(p.name))
    total = len(sorted_files)
    combined_pages = []

    for idx, f in enumerate(sorted_files, 1):
        progress((idx - 1) / total, desc=f"Đang bóc tách trang [{idx}/{total}]: {f.name}")
        try:
            img_bgr, meta = cleaner.load_and_preprocess(f)
            res = engine.recognize(img_bgr)
            text = res["full_text"].strip()
            combined_pages.append(f"=== [Trang {idx}/{total}] {f.name} ===\n\n{text}")
        except Exception as err:
            combined_pages.append(f"=== [Trang {idx}/{total}] {f.name} (LỖI) ===\n\nLỗi: {str(err)}")

    full_text = "\n\n".join(combined_pages)
    elapsed = round(time.time() - t0, 2)
    info = f"✔ Hoàn thành {total} trang trong {elapsed} giây (trung bình {round(elapsed/total, 1)}s/trang)."

    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp_txt.write(full_text)
    tmp_txt.close()

    return full_text, info, tmp_txt.name


def process_gdrive_url(
    gdrive_link: str,
    custom_api_key: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    Download a folder or archive from Google Drive and perform OCR on its pages.
    """
    link = (gdrive_link or "").strip()
    if not link or not is_gdrive_url(link):
        return "⚠️ Đường dẫn không hợp lệ. Vui lòng nhập link chia sẻ Google Drive.", "Link không hợp lệ", None

    api_key = (custom_api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ Chưa có Gemini API Key. Vui lòng nhập API Key vào ô cấu hình.", "Thiếu API Key", None

    temp_root = Path(tempfile.gettempdir()) / f"hf_ocr_{int(time.time())}"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.1, desc="Đang phân tích và tải dữ liệu từ Google Drive...")
        book_name, target_dir, count = download_and_extract_gdrive(
            url=link,
            input_root=temp_root
        )

        progress(0.3, desc=f"Đã tải xong cuốn '{book_name}' ({count} trang). Đang chuẩn bị OCR...")

        engine = GeminiVisionEngine(api_key=api_key)
        cleaner = ImageCleaner(max_dimension=2560, auto_orient=True)

        pages = sorted(list(target_dir.glob("*.heic")) + list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png")),
                       key=lambda p: natural_sort_key(p.name))

        combined_pages = []
        total = len(pages)
        t0 = time.time()

        for idx, p in enumerate(pages, 1):
            progress(0.3 + (idx / total) * 0.65, desc=f"Đang OCR trang [{idx}/{total}]...")
            try:
                img_bgr, meta = cleaner.load_and_preprocess(p)
                res = engine.recognize(img_bgr)
                combined_pages.append(f"=== [Trang {idx}/{total}] {p.name} ===\n\n{res['full_text'].strip()}")
            except Exception as e:
                combined_pages.append(f"=== [Trang {idx}/{total}] {p.name} ===\n\nLỗi: {str(e)}")

        full_text = "\n\n".join(combined_pages)
        elapsed = round(time.time() - t0, 1)
        info = f"✔ Hoàn thành cuốn sách '{book_name}' ({count} trang) trong {elapsed}s!"

        tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{book_name}.txt", mode="w", encoding="utf-8")
        tmp_txt.write(full_text)
        tmp_txt.close()

        return full_text, info, tmp_txt.name

    except Exception as e:
        return f"❌ Lỗi tải hoặc xử lý Google Drive: {str(e)}", f"Lỗi: {str(e)}", None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# Custom CSS for Modern UI
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.hero-header {
    text-align: center;
    padding: 24px 10px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #6366f1 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 24px;
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
    color: #ffffff !important;
}
.hero-header p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 0;
}
.info-card {
    border-radius: 8px;
    padding: 12px 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
"""

with gr.Blocks(title="Book OCR Studio AI", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero-header">
        <h1>📚 Book OCR Studio AI</h1>
        <p>Số hóa sách cổ, tài liệu lưu trữ tiếng Việt với độ chính xác 100% bằng Google Gemini Vision AI</p>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: SINGLE / MULTI-IMAGE OCR
        with gr.Tab("📄 Trích xuất Trang sách (Ảnh lẻ / Nhiều ảnh)"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_images = gr.File(
                        label="Tải lên ảnh trang sách (Hỗ trợ JPG, PNG, WEBP, iPhone HEIC/HEIF)",
                        file_count="multiple",
                        file_types=["image", ".heic", ".heif"]
                    )
                    with gr.Accordion("⚙️ Tùy chọn nâng cao & API Key", open=False):
                        split_double = gr.Checkbox(
                            label="Tự động phân tách trang đôi (Double-page split: Trang Trái / Phải)",
                            value=False
                        )
                        auto_orient = gr.Checkbox(
                            label="Tự động căn chỉnh xoay ảnh nghiêng",
                            value=True
                        )
                        custom_key = gr.Textbox(
                            label="Gemini API Key (Tùy chọn - mặc định dùng Secret của hệ thống)",
                            placeholder="AIzaSy...",
                            type="password"
                        )
                    btn_ocr = gr.Button("🚀 Bắt đầu OCR Văn Bản", variant="primary", size="lg")

                with gr.Column(scale=7):
                    txt_output = gr.Textbox(
                        label="📖 Văn bản tiếng Việt trích xuất được",
                        lines=18,
                        placeholder="Nội dung sách sau khi bóc tách sẽ hiển thị ở đây..."
                    )
                    status_info = gr.Markdown("Trạng thái: Sẵn sàng")
                    file_download = gr.File(label="📥 Tải về file kết quả (.TXT)")

            def _handle_ocr(files, split_val, key_val, orient_val):
                if not files:
                    return "⚠️ Vui lòng chọn ít nhất 1 file ảnh.", "Chưa có ảnh.", None
                if len(files) == 1:
                    return ocr_single_image(files[0], split_val, orient_val, key_val)
                else:
                    return ocr_multiple_images(files, split_val, key_val)

            btn_ocr.click(
                fn=_handle_ocr,
                inputs=[input_images, split_double, custom_key, auto_orient],
                outputs=[txt_output, status_info, file_download]
            )

        # TAB 2: GOOGLE DRIVE & ARCHIVE
        with gr.Tab("📥 Tải trực tiếp từ Google Drive"):
            with gr.Row():
                with gr.Column(scale=5):
                    gdrive_input = gr.Textbox(
                        label="Đường dẫn Google Drive (Link thư mục hoặc file nén .rar, .zip, .7z)",
                        placeholder="https://drive.google.com/drive/folders/1bwmWslmBIaW1dbreRlxjhARFZSo9cjQm...",
                        lines=2
                    )
                    gdrive_key = gr.Textbox(
                        label="Gemini API Key (Tùy chọn)",
                        placeholder="AIzaSy... (để trống nếu dùng Secret của Space)",
                        type="password"
                    )
                    btn_gdrive = gr.Button("⚡ Kéo dữ liệu & Tự động OCR", variant="primary", size="lg")
                    gr.Markdown("""
                    > **Lưu ý:** Đảm bảo link Google Drive đã được bật quyền chia sẻ:  
                    > *"Bất kỳ ai có đường liên kết đều có thể xem" (Anyone with the link can view).*
                    """)

                with gr.Column(scale=7):
                    gdrive_output = gr.Textbox(
                        label="📖 Nội dung toàn bộ cuốn sách",
                        lines=18,
                        placeholder="Văn bản toàn cuốn sách sau khi tải và bóc tách sẽ hiển thị tại đây..."
                    )
                    gdrive_status = gr.Markdown("Trạng thái: Chờ nhập link")
                    gdrive_download = gr.File(label="📥 Tải trọn vẹn sách (.TXT)")

            btn_gdrive.click(
                fn=process_gdrive_url,
                inputs=[gdrive_input, gdrive_key],
                outputs=[gdrive_output, gdrive_status, gdrive_download]
            )

        # TAB 3: GUIDE & ABOUT
        with gr.Tab("ℹ️ Hướng dẫn & Giới thiệu"):
            gr.Markdown("""
            ### 🌟 Về Book OCR Studio AI
            Hệ thống nhận dạng ký tự quang học (OCR) chuyên sâu cho **sách, tài liệu lịch sử, luận văn tiếng Việt** với các tính năng vượt trội:
            - **Độ chính xác 100% tiếng Việt:** Tận dụng thị giác AI đa phương thức của Google Gemini (`gemini-2.5-flash`).
            - **Tự động cấu trúc:** Giữ nguyên tiêu đề in hoa, số La Mã, thứ tự mục lục và định dạng ngắt đoạn.
            - **Xử lý ảnh iPhone HEIC/HEIF:** Đọc trực tiếp các bức ảnh chụp từ điện thoại mà không cần đổi đuôi.
            - **Tải Google Drive song song 4 luồng:** Tải trọn vẹn cả thư mục 300+ trang ảnh chỉ trong vài phút.

            ### 🔑 Hướng dẫn lấy Gemini API Key miễn phí:
            1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey).
            2. Đăng nhập bằng tài khoản Google bất kỳ và bấm **Create API key**.
            3. Dán key vào ô cấu hình ở trên để sử dụng không giới hạn!
            """)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
