import sys
import io

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from src.config import AppConfig
from src.db.state_tracker import StateTracker
from src.utils.file_utils import scan_image_files, get_book_folders, natural_sort_key
from src.pipeline.dispatcher import Dispatcher

app = typer.Typer(
    name="OCR Batch Pipeline",
    help="Hệ thống chuyển đổi số lượng lớn ảnh sang văn bản (Image to Text) hiệu năng cao theo từng cuốn sách.",
    add_completion=False
)
console = Console(legacy_windows=False)



def _load_app_config(config_path: str = "configs/config.yaml") -> AppConfig:
    cfg = AppConfig.load_from_yaml(config_path)
    # Ensure standard directories exist
    Path(cfg.paths.input_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@app.command()
def scan(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Chỉ định quét một cuốn sách cụ thể"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Quét thư mục ảnh đầu vào (hoặc 1 cuốn sách) và đăng ký các trang mới vào cơ sở dữ liệu.
    """
    cfg = _load_app_config(config)
    tracker = StateTracker(cfg.paths.db_path)

    target_desc = f"sách '{book}'" if book else f"toàn bộ thư mục {cfg.paths.input_dir}"
    console.print(f"[bold cyan]Đang quét ảnh đầu vào:[/bold cyan] {target_desc}")
    scanned_items = list(scan_image_files(cfg.paths.input_dir, book_name=book))
    new_count = tracker.register_files(scanned_items)

    console.print(f"-> Tìm thấy [bold green]{len(scanned_items)}[/bold green] file ảnh trang sách.")
    console.print(f"-> Đăng ký mới [bold green]{new_count}[/bold green] file vào danh sách chờ xử lý.")


@app.command()
def run(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Chỉ định chạy OCR riêng cho cuốn sách cụ thể"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Giới hạn số lượng ảnh xử lý trong lượt này"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Ghi đè số lượng tiến trình worker song song"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Tự động quét ảnh mới và thực hiện OCR toàn bộ hoặc theo cuốn sách.
    """
    cfg = _load_app_config(config)
    if workers:
        cfg.system.num_workers = workers

    tracker = StateTracker(cfg.paths.db_path)

    # 1. Quick scan to register any new files
    scanned_items = list(scan_image_files(cfg.paths.input_dir, book_name=book))
    new_count = tracker.register_files(scanned_items)
    if new_count > 0:
        console.print(f"[dim]Đã thêm {new_count} file ảnh mới vào hàng đợi.[/dim]")

    # 2. Dispatch processing
    dispatcher = Dispatcher(cfg)
    dispatcher.run(max_limit=limit, book_name=book)


@app.command()
def resume(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Chỉ định tiếp tục cuốn sách cụ thể"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Giới hạn số lượng ảnh xử lý"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Tiếp tục xử lý các tác vụ đang dở hoặc chưa hoàn thành.
    """
    cfg = _load_app_config(config)
    dispatcher = Dispatcher(cfg)
    dispatcher.run(max_limit=limit, book_name=book)


@app.command()
def retry_failed(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Chỉ định thử lại các trang lỗi của cuốn sách cụ thể"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Đặt lại trạng thái các file bị lỗi (FAILED) về PENDING và xử lý lại.
    """
    cfg = _load_app_config(config)
    tracker = StateTracker(cfg.paths.db_path)
    count = tracker.retry_failed_tasks(book_name=book)

    if count == 0:
        msg = f"Không có file lỗi nào trong sách '{book}'." if book else "Không có file nào bị lỗi cần thử lại."
        console.print(f"[green]{msg}[/green]")
        return

    console.print(f"[yellow]Đã đặt lại {count} file lỗi về trạng thái PENDING. Bắt đầu xử lý lại...[/yellow]")
    dispatcher = Dispatcher(cfg)
    dispatcher.run(book_name=book)


@app.command()
def books(
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Liệt kê danh sách tất cả các cuốn sách (thư mục con) và tiến độ OCR của từng cuốn.
    """
    cfg = _load_app_config(config)
    tracker = StateTracker(cfg.paths.db_path)

    # Make sure any un-scanned books are discovered
    scanned_items = list(scan_image_files(cfg.paths.input_dir))
    if scanned_items:
        tracker.register_files(scanned_items)

    book_stats = tracker.get_book_statistics()

    table = Table(title="📚 DANH SÁCH CÁC CUỐN SÁCH & TIẾN ĐỘ OCR", title_justify="center")
    table.add_column("STT", justify="center", style="dim")
    table.add_column("Tên cuốn sách (Folder)", style="bold cyan")
    table.add_column("Tổng trang", justify="right")
    table.add_column("Đã xong", justify="right", style="green")
    table.add_column("Đang chờ", justify="right", style="yellow")
    table.add_column("Lỗi", justify="right", style="red")
    table.add_column("Tiến độ", justify="center", style="bold")
    table.add_column("Trạng thái", justify="center")

    if not book_stats:
        console.print("[yellow]Chưa có thư mục sách nào trong 'data/input'. Hãy tạo thư mục tên sách và cho ảnh vào.[/yellow]")
        return

    for idx, b in enumerate(book_stats, 1):
        pct = b["percent"]
        if pct == 100.0:
            status_badge = "[bold green]✔ HOÀN TẤT[/bold green]"
            pct_style = f"[green]{pct}%[/green]"
        elif pct > 0:
            status_badge = "[bold yellow]⏳ ĐANG OCR[/bold yellow]"
            pct_style = f"[yellow]{pct}%[/yellow]"
        else:
            status_badge = "[dim]CHƯA CHẠY[/dim]"
            pct_style = f"[dim]{pct}%[/dim]"

        table.add_row(
            str(idx),
            b["book_name"],
            str(b["total"]),
            str(b["done"]),
            str(b["pending"]),
            str(b["failed"]),
            pct_style,
            status_badge
        )

    console.print(table)


@app.command()
def merge_book(
    book_name: str = typer.Argument(..., help="Tên thư mục cuốn sách cần gộp file text"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Đường dẫn file text đầu ra"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Ghép tất cả các trang văn bản (.txt) của 1 cuốn sách thành 1 file text duy nhất theo đúng thứ tự trang.
    """
    cfg = _load_app_config(config)
    book_output_dir = Path(cfg.paths.output_dir) / book_name

    if not book_output_dir.is_dir():
        console.print(f"[bold red]Lỗi:[/bold red] Không tìm thấy thư mục kết quả của sách: {book_output_dir}")
        return

    txt_files = [f for f in book_output_dir.rglob("*.txt") if f.is_file()]
    if not txt_files:
        console.print(f"[bold red]Lỗi:[/bold red] Chưa có file .txt nào trong {book_output_dir}. Hãy chạy OCR cuốn sách này trước!")
        return

    # Sort naturally so pages are strictly in 1, 2, ..., 9, 10 order
    txt_files = sorted(txt_files, key=lambda p: natural_sort_key(p.name))

    out_path = Path(output_file) if output_file else Path(cfg.paths.output_dir) / f"{book_name}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, f in enumerate(txt_files, 1):
            content = f.read_text(encoding="utf-8", errors="replace").strip()
            total_chars += len(content)
            out.write(f"=== [Trang {idx}/{len(txt_files)}] {f.name} ===\n\n")
            out.write(content)
            out.write("\n\n")

    console.print(f"[bold green]✔ Ghép thành công sách:[/bold green] [bold cyan]{book_name}[/bold cyan]")
    console.print(f"-> Tổng số trang gộp:  [bold]{len(txt_files)}[/bold] trang")
    console.print(f"-> Tổng ký tự:         [bold]{total_chars:,}[/bold]")
    console.print(f"-> File đầu ra:        [bold yellow]{out_path.resolve()}[/bold yellow] ({out_path.stat().st_size / 1024:.1f} KB)")


@app.command()
def merge_all(
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Tự động ghép file text hoàn chỉnh cho tất cả các cuốn sách đã có kết quả OCR.
    """
    cfg = _load_app_config(config)
    output_path = Path(cfg.paths.output_dir)
    book_dirs = [d for d in output_path.iterdir() if d.is_dir()]

    if not book_dirs:
        console.print(f"[yellow]Chưa có thư mục sách nào trong {output_path}[/yellow]")
        return

    for d in sorted(book_dirs, key=lambda p: natural_sort_key(p.name)):
        merge_book(book_name=d.name, output_file=None, config=config)


@app.command()
def status(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Lọc thống kê theo cuốn sách cụ thể"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Xem thống kê tổng quan tiến độ xử lý dữ liệu (toàn bộ hoặc theo sách).
    """
    cfg = _load_app_config(config)
    tracker = StateTracker(cfg.paths.db_path)
    s = tracker.get_statistics(book_name=book)

    title = f"📊 THỐNG KÊ TIẾN ĐỘ OCR [{book}]" if book else "📊 THỐNG KÊ TIẾN ĐỘ XỬ LÝ OCR"
    table = Table(title=title, title_justify="center")
    table.add_column("Chỉ số", style="cyan", no_wrap=True)
    table.add_column("Giá trị", style="bold white")

    table.add_row("📁 Tổng số ảnh", str(s["total"]))
    table.add_row("✅ Đã hoàn thành (DONE)", f"[green]{s['done']}[/green]")
    table.add_row("⏳ Đang chờ (PENDING)", f"[yellow]{s['pending']}[/yellow]")
    table.add_row("⚙️ Đang chạy (IN_PROGRESS)", str(s["in_progress"]))
    table.add_row("❌ Thất bại (FAILED)", f"[red]{s['failed']}[/red]")
    table.add_row("📈 Tỉ lệ hoàn thành", f"[bold cyan]{s['percent_complete']}%[/bold cyan]")
    table.add_row("⚡ Thời gian trung bình / ảnh", f"{s['avg_time_ms']} ms")
    table.add_row("🎯 Độ tin cậy trung bình", f"{s['avg_confidence']}")

    console.print(table)


@app.command()
def export_failed(
    output_path: str = typer.Option("logs/failed_tasks.csv", "--output", "-o", help="Đường dẫn lưu file CSV báo cáo lỗi"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Xuất danh sách các file bị lỗi kèm thông báo lỗi chi tiết ra file CSV.
    """
    import csv
    cfg = _load_app_config(config)
    tracker = StateTracker(cfg.paths.db_path)
    
    with tracker._get_connection() as conn:
        rows = conn.execute(
            "SELECT file_path, file_size, error_msg, updated_at FROM tasks WHERE status = 'FAILED'"
        ).fetchall()

    if not rows:
        console.print("[green]Không có file nào bị lỗi![/green]")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "file_size_bytes", "error_message", "timestamp"])
        for r in rows:
            writer.writerow([r["file_path"], r["file_size"], r["error_msg"], r["updated_at"]])

@app.command()
def test_image(
    image_path: str = typer.Argument(..., help="Đường dẫn tới file ảnh cần kiểm tra"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Chỉ định engine (gemini, vietocr, rapidocr, easyocr)"),
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Đường dẫn file cấu hình YAML")
):
    """
    Chạy thử nghiệm OCR trên một bức ảnh cụ thể và in trực tiếp nội dung trích xuất ra màn hình.
    """
    cfg = _load_app_config(config)
    if engine:
        cfg.ocr.engine = engine

    img_p = Path(image_path)
    if not img_p.exists():
        console.print(f"[bold red]Lỗi: Không tìm thấy file ảnh tại {image_path}[/bold red]")
        return

    console.print(f"[bold cyan]Đang xử lý thử nghiệm ảnh:[/bold cyan] {img_p.name} bằng engine [yellow]{cfg.ocr.engine}[/yellow]...")

    cfg_dict = cfg.model_dump()
    from src.pipeline.worker import _get_engine_and_cleaner
    eng, cleaner = _get_engine_and_cleaner(cfg_dict)

    img_bgr, meta = cleaner.load_and_preprocess(img_p)
    console.print(f"[dim]Thông số ảnh: Gốc: {meta['orig_width']}x{meta['orig_height']} -> Xử lý: {meta['processed_width']}x{meta['processed_height']}, Xoay nghiêng: {meta['rotated_sideways']}, Mở sách 2 trang: {meta['is_double_page']}[/dim]")

    split_double = cfg.preprocessing.split_double_pages
    if split_double and meta.get("is_double_page", False):
        pages = cleaner.split_pages(img_bgr)
        for idx, p in enumerate(pages):
            label = "TRANG TRÁI" if idx == 0 else "TRANG PHẢI"
            console.print(f"\n[bold green]=== [{label}] ===[/bold green]")
            res = eng.recognize(p)
            console.print(res["full_text"])
    else:
        res = eng.recognize(img_bgr)
        console.print("\n[bold green]=== KẾT QUẢ TRÍCH XUẤT VĂN BẢN ===[/bold green]")
        console.print(res["full_text"])
        console.print(f"\n[dim]Thời gian xử lý: {res['elapse_ms']} ms | Độ tin cậy: {res['avg_confidence']}[/dim]")


if __name__ == "__main__":
    app()

