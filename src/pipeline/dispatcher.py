import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn
)
from rich.console import Console

from src.config import AppConfig
from src.db.state_tracker import StateTracker
from src.writers.txt_writer import TxtWriter
from src.writers.jsonl_writer import JsonlWriter
from src.pipeline.worker import process_image_task

console = Console(legacy_windows=False)



class Dispatcher:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracker = StateTracker(config.paths.db_path)
        self.input_dir = Path(config.paths.input_dir).resolve()
        self.output_dir = Path(config.paths.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup writers
        self.writers = []
        fmt = self.config.output.format.lower()
        if fmt in ("txt", "both"):
            self.writers.append(
                TxtWriter(
                    self.output_dir,
                    mirror_structure=self.config.output.mirror_folder_structure
                )
            )
        if fmt in ("jsonl", "both"):
            self.writers.append(
                JsonlWriter(self.output_dir / "results.jsonl")
            )

    def run(
        self,
        max_limit: Optional[int] = None,
        book_name: Optional[str] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Execute OCR processing on all pending tasks.
        If book_name is provided, only process pages for that specific book folder.
        Set show_progress=False when called from a non-console context (web thread).
        """
        # 1. Reset any lingering tasks from previous abrupt shutdowns
        hanging = self.tracker.reset_hanging_tasks(book_name=book_name)
        if hanging > 0 and show_progress:
            console.print(f"[yellow]Đã đặt lại {hanging} tác vụ bị gián đoạn về trạng thái PENDING.[/yellow]")

        # 2. Get pending count
        stats = self.tracker.get_statistics(book_name=book_name)
        total_pending = stats["pending"]
        if total_pending == 0:
            if show_progress:
                msg = f"Không có ảnh nào cần xử lý cho sách '{book_name}'." if book_name else "Không có ảnh nào cần xử lý. Tất cả đều đã hoàn thành!"
                console.print(f"[green]{msg}[/green]")
            return

        limit_to_process = min(total_pending, max_limit) if max_limit else total_pending
        num_workers = self.config.system.num_workers
        batch_size = self.config.system.batch_size

        if show_progress:
            book_info = f" | Cuốn sách: [bold magenta]{book_name}[/bold magenta]" if book_name else ""
            console.print(
                f"[bold cyan]Bắt đầu xử lý OCR:[/bold cyan] "
                f"Tổng số [green]{limit_to_process}[/green] ảnh{book_info} | "
                f"Số luồng song song: [yellow]{num_workers}[/yellow] | "
                f"Batch: [yellow]{batch_size}[/yellow]"
            )

        cfg_dict = self.config.model_dump()

        # Rich Progress Setup
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("• Tốc độ: [cyan]{task.fields[speed]:.1f} img/s[/cyan]"),
            TimeRemainingColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            task_id = progress.add_task(
                f"Đang nhận diện văn bản{f' [{book_name}]' if book_name else ''}...",
                total=limit_to_process,
                speed=0.0
            )

            processed_count = 0
            start_overall_time = time.perf_counter()

            # ThreadPoolExecutor: stable on Windows with PyTorch (no pickling, shared memory)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                try:
                    while processed_count < limit_to_process:
                        # Fetch next batch of pending tasks
                        current_limit = min(batch_size, limit_to_process - processed_count)
                        pending_items = self.tracker.claim_pending_tasks(limit=current_limit, book_name=book_name)
                        if not pending_items:
                            break

                        file_paths = [item["file_path"] for item in pending_items]

                        # Submit chunk of tasks — config_dict passed per-task for thread-local init
                        futures = [
                            executor.submit(process_image_task, (str(self.input_dir), fp, cfg_dict))
                            for fp in file_paths
                        ]

                        batch_results: List[Dict[str, Any]] = []

                        for future in as_completed(futures):
                            res = future.result()
                            batch_results.append(res)

                            # If success, write output immediately
                            if res["success"]:
                                for w in self.writers:
                                    w.write(res["file_path"], res)

                            processed_count += 1
                            elapsed = time.perf_counter() - start_overall_time
                            speed = (processed_count / elapsed) if elapsed > 0 else 0.0

                            progress.update(
                                task_id,
                                advance=1,
                                speed=speed
                            )

                        # Commit batch results to SQLite
                        self.tracker.batch_update_results(batch_results)

                except KeyboardInterrupt:
                    console.print("\n[bold yellow]Đã nhận tín hiệu dừng (Ctrl+C). Đang hoàn tất lưu dữ liệu...[/bold yellow]")
                finally:
                    for w in self.writers:
                        w.close()

        if not show_progress:
            return

        # Summary report
        final_stats = self.tracker.get_statistics(book_name=book_name)
        target_title = f" [{book_name}]" if book_name else ""
        console.print(f"\n[bold green]=== BÁO CÁO TIẾN ĐỘ HOÀN TẤT{target_title} ===[/bold green]")
        console.print(f"Tổng số ảnh trong hệ thống: [bold]{final_stats['total']}[/bold]")
        console.print(f"Thành công (DONE):           [bold green]{final_stats['done']}[/bold green]")
        console.print(f"Thất bại (FAILED):          [bold red]{final_stats['failed']}[/bold red]")
        console.print(f"Còn lại (PENDING):          [bold yellow]{final_stats['pending']}[/bold yellow]")
        console.print(f"Tỉ lệ hoàn thành:           [bold cyan]{final_stats['percent_complete']}%[/bold cyan]")
        if final_stats['avg_time_ms'] > 0:
            console.print(f"Thời gian TB mỗi ảnh:       [bold]{final_stats['avg_time_ms']} ms[/bold]")

