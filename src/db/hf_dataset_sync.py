import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from huggingface_hub import HfApi, get_token, hf_hub_download

from .book_repository import BookRepository, slugify

DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO", "cuongblowurmind/book-ocr-library")


class HFDatasetSync:
    """
    Manages persistent synchronization of OCR books, text outputs, thumbnails,
    and metadata with a private Hugging Face Dataset repository.
    Ensures zero data loss across Hugging Face Space restarts, rebuilds, and deployments.
    """

    def __init__(
        self,
        repo_id: str = DATASET_REPO_ID,
        output_dir: str | Path = "data/output",
        covers_dir: str | Path = "data/covers"
    ):
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.covers_dir = Path(covers_dir)
        self.token = os.environ.get("HF_TOKEN") or get_token()
        self.api = HfApi(token=self.token) if self.token else None

    def is_configured(self) -> bool:
        return bool(self.api and self.token and self.repo_id)

    def push_book(self, slug: str, book_repo: Optional[BookRepository] = None) -> bool:
        """
        Uploads a book's .txt file, cover image, and metadata JSON to the dataset under /{slug}/.
        """
        if not self.is_configured():
            print("[HFDatasetSync] HF_TOKEN or repo not configured. Skipping cloud sync.")
            return False

        book_repo = book_repo or BookRepository()
        book = book_repo.get_book_by_slug(slug)
        if not book:
            print(f"[HFDatasetSync] Book not found in database: {slug}")
            return False

        title = book["title"]
        text_file = self.output_dir / f"{title}.txt"
        cover_file = self.covers_dir / f"{slug}.jpg"

        try:
            # 1. Upload text file if exists
            if text_file.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(text_file),
                    path_in_repo=f"{slug}/{slug}.txt",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Sync book text: {title}"
                )

            # 2. Upload cover if exists
            if cover_file.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(cover_file),
                    path_in_repo=f"{slug}/cover.jpg",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Sync book cover: {title}"
                )

            # 3. Upload metadata.json
            meta_json = json.dumps(book, ensure_ascii=False, indent=2).encode("utf-8")
            self.api.upload_file(
                path_or_fileobj=meta_json,
                path_in_repo=f"{slug}/metadata.json",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Sync book metadata: {title}"
            )

            # 4. Update root library_manifest.json
            self.update_manifest(book_repo)
            print(f"[HFDatasetSync] Successfully synced '{title}' to Hugging Face Dataset!")
            return True

        except Exception as e:
            print(f"[HFDatasetSync] Error syncing '{title}' to dataset: {e}")
            return False

    def push_book_async(self, slug: str):
        """Run push_book in a background thread."""
        t = threading.Thread(target=self.push_book, args=(slug,), daemon=True)
        t.start()

    def update_manifest(self, book_repo: Optional[BookRepository] = None):
        """Upload all current books list as library_manifest.json on dataset root."""
        if not self.is_configured():
            return
        book_repo = book_repo or BookRepository()
        books_data = book_repo.list_books(limit=1000)
        manifest_bytes = json.dumps(books_data["books"], ensure_ascii=False, indent=2).encode("utf-8")
        try:
            self.api.upload_file(
                path_or_fileobj=manifest_bytes,
                path_in_repo="library_manifest.json",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Update library manifest"
            )
        except Exception as e:
            print(f"[HFDatasetSync] Error updating library_manifest.json: {e}")

    def pull_manifest_and_restore(self, book_repo: Optional[BookRepository] = None) -> int:
        """
        Pull library_manifest.json from Dataset, restore books into SQLite database,
        and download covers to data/covers/{slug}.jpg.
        Called upon startup to ensure full library persistence.
        """
        if not self.is_configured():
            return 0

        book_repo = book_repo or BookRepository()
        self.covers_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            manifest_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename="library_manifest.json",
                token=self.token
            )
            with open(manifest_path, "r", encoding="utf-8") as f:
                books = json.load(f)

            restored = 0
            for b in books:
                slug = b.get("slug") or slugify(b["title"])
                cover_dest = self.covers_dir / f"{slug}.jpg"

                # Download cover if missing
                cover_url = f"/covers/{slug}.jpg"
                if not cover_dest.is_file():
                    try:
                        remote_cover = hf_hub_download(
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            filename=f"{slug}/cover.jpg",
                            token=self.token
                        )
                        import shutil
                        shutil.copy(remote_cover, cover_dest)
                    except Exception:
                        pass

                book_repo.upsert_book(
                    title=b["title"],
                    slug=slug,
                    category=b.get("category", "Tài liệu tổng hợp"),
                    tags=b.get("tags", ""),
                    total_pages=b.get("total_pages", 0),
                    word_count=b.get("word_count", 0),
                    file_size_bytes=b.get("file_size_bytes", 0),
                    source_type=b.get("source_type", "upload"),
                    source_id=b.get("source_id", ""),
                    cover_image=cover_url,
                    summary=b.get("summary", ""),
                    is_public=b.get("is_public", 1),
                    status=b.get("status", "COMPLETED"),
                    artifact_code=b.get("artifact_code", ""),
                    period_era=b.get("period_era", ""),
                    provenance=b.get("provenance", ""),
                    material_condition=b.get("material_condition", ""),
                    curator_notes=b.get("curator_notes", ""),
                    exhibition_hall=b.get("exhibition_hall", "Khu Trưng Bày Chung"),
                    highlights_json=b.get("highlights_json", "[]")
                )
                restored += 1

            print(f"[HFDatasetSync] Successfully restored {restored} books from Hugging Face Dataset!")
            return restored

        except Exception as e:
            print(f"[HFDatasetSync] Manifest download error or repo empty: {e}")
            return 0

    def ensure_book_text_file(self, slug: str, title: str) -> Optional[Path]:
        """
        Ensures data/output/{title}.txt exists locally. If not found, downloads it
        on-demand from the dataset repo.
        """
        text_file = self.output_dir / f"{title}.txt"
        if text_file.is_file() and text_file.stat().st_size > 0:
            return text_file

        if not self.is_configured():
            return None

        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"{slug}/{slug}.txt",
                token=self.token
            )
            import shutil
            self.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, text_file)
            return text_file
        except Exception as e:
            print(f"[HFDatasetSync] Could not download text file for {slug}: {e}")
            return None

    def ensure_book_cover(self, slug: str) -> Optional[Path]:
        """Ensures data/covers/{slug}.jpg exists locally."""
        cover_dest = self.covers_dir / f"{slug}.jpg"
        if cover_dest.is_file() and cover_dest.stat().st_size > 0:
            return cover_dest

        if not self.is_configured():
            return None

        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"{slug}/cover.jpg",
                token=self.token
            )
            import shutil
            self.covers_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, cover_dest)
            return cover_dest
        except Exception:
            return None
