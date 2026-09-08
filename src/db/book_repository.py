import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def slugify(text: str) -> str:
    """Convert Vietnamese title to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[đĐ]", "d", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "untitled"


def auto_categorize(title: str) -> str:
    """Heuristic categorization based on book title."""
    lower = title.lower()
    if any(k in lower for k in ["luận án", "tiến sĩ", "nghiên cứu", "khảo sát"]):
        return "Nghiên cứu / Luận án"
    if any(k in lower for k in ["giáo trình", "bài giảng", "học liệu", "đại học"]):
        return "Giáo trình - Học liệu"
    if any(k in lower for k in ["lịch sử", "địa chí", "vùng đất", "dân cư", "yên mỹ"]):
        return "Lịch sử - Địa chí"
    if any(k in lower for k in ["thờ", "tín ngưỡng", "chùa", "đền", "tâm linh", "chử đồng tử"]):
        return "Văn hóa - Tín ngưỡng"
    if any(k in lower for k in ["tiểu thuyết", "truyện", "thơ", "văn học"]):
        return "Văn học nghệ thuật"
    return "Tài liệu tổng hợp"


class BookRepository:
    def __init__(
        self,
        db_path: str | Path = "data/tracker.db",
        input_dir: str | Path = "data/input",
        output_dir: str | Path = "data/output",
        covers_dir: str | Path = "data/covers",
        cache_dir: str | Path = "data/page_cache"
    ):
        self.db_path = Path(db_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.covers_dir = Path(covers_dir)
        self.cache_dir = Path(cache_dir)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.covers_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT DEFAULT 'Tài liệu tổng hợp',
                    tags TEXT DEFAULT '',
                    total_pages INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0,
                    file_size_bytes INTEGER DEFAULT 0,
                    source_type TEXT DEFAULT 'upload',
                    source_id TEXT DEFAULT '',
                    cover_image TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    is_public INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'COMPLETED',
                    artifact_code TEXT DEFAULT '',
                    period_era TEXT DEFAULT '',
                    provenance TEXT DEFAULT '',
                    material_condition TEXT DEFAULT '',
                    curator_notes TEXT DEFAULT '',
                    exhibition_hall TEXT DEFAULT 'Khu Trưng Bày Chung',
                    highlights_json TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_slug ON books(slug);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_category ON books(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_source ON books(source_type, source_id);")

            # Check and run migrations if columns are missing
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(books);").fetchall()}
            new_cols = [
                ("artifact_code", "TEXT DEFAULT ''"),
                ("period_era", "TEXT DEFAULT ''"),
                ("provenance", "TEXT DEFAULT ''"),
                ("material_condition", "TEXT DEFAULT ''"),
                ("curator_notes", "TEXT DEFAULT ''"),
                ("exhibition_hall", "TEXT DEFAULT 'Khu Trưng Bày Chung'"),
                ("highlights_json", "TEXT DEFAULT '[]'"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE books ADD COLUMN {col_name} {col_type};")
            conn.commit()

    def generate_cover_thumbnail(self, book_name: str, slug: str) -> str:
        """
        Extract the first page image of the book, resize to 400x600 thumbnail,
        and save to data/covers/{slug}.jpg.
        """
        target_cover = self.covers_dir / f"{slug}.jpg"
        if target_cover.is_file() and target_cover.stat().st_size > 0:
            return f"/covers/{slug}.jpg"

        book_input = self.input_dir / book_name
        if not book_input.is_dir():
            return ""

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}
        images = sorted([
            f for f in book_input.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith(".")
        ])

        if not images:
            return ""

        first_img = images[0]
        try:
            with Image.open(first_img) as im:
                im = im.convert("RGB")
                im.thumbnail((400, 600), Image.Resampling.LANCZOS)
                im.save(target_cover, format="JPEG", quality=85, optimize=True)
            return f"/covers/{slug}.jpg"
        except Exception as e:
            print(f"[BookRepository] Error generating cover for {book_name}: {e}")
            return ""

    def upsert_book(
        self,
        title: str,
        slug: Optional[str] = None,
        category: Optional[str] = None,
        tags: str = "",
        total_pages: int = 0,
        word_count: int = 0,
        file_size_bytes: int = 0,
        source_type: str = "upload",
        source_id: str = "",
        cover_image: str = "",
        summary: str = "",
        is_public: int = 1,
        status: str = "COMPLETED",
        artifact_code: str = "",
        period_era: str = "",
        provenance: str = "",
        material_condition: str = "",
        curator_notes: str = "",
        exhibition_hall: str = "Khu Trưng Bày Chung",
        highlights_json: str = "[]"
    ) -> Dict[str, Any]:
        slug = slug or slugify(title)
        category = category or auto_categorize(title)
        if not cover_image:
            cover_image = self.generate_cover_thumbnail(title, slug)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO books (
                    slug, title, category, tags, total_pages, word_count,
                    file_size_bytes, source_type, source_id, cover_image,
                    summary, is_public, status,
                    artifact_code, period_era, provenance,
                    material_condition, curator_notes, exhibition_hall,
                    highlights_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(slug) DO UPDATE SET
                    title = excluded.title,
                    total_pages = CASE WHEN excluded.total_pages > 0 THEN excluded.total_pages ELSE books.total_pages END,
                    word_count = CASE WHEN excluded.word_count > 0 THEN excluded.word_count ELSE books.word_count END,
                    file_size_bytes = CASE WHEN excluded.file_size_bytes > 0 THEN excluded.file_size_bytes ELSE books.file_size_bytes END,
                    cover_image = CASE WHEN excluded.cover_image != '' THEN excluded.cover_image ELSE books.cover_image END,
                    status = excluded.status,
                    artifact_code = CASE WHEN excluded.artifact_code != '' THEN excluded.artifact_code ELSE books.artifact_code END,
                    period_era = CASE WHEN excluded.period_era != '' THEN excluded.period_era ELSE books.period_era END,
                    provenance = CASE WHEN excluded.provenance != '' THEN excluded.provenance ELSE books.provenance END,
                    material_condition = CASE WHEN excluded.material_condition != '' THEN excluded.material_condition ELSE books.material_condition END,
                    curator_notes = CASE WHEN excluded.curator_notes != '' THEN excluded.curator_notes ELSE books.curator_notes END,
                    exhibition_hall = CASE WHEN excluded.exhibition_hall != 'Khu Trưng Bày Chung' THEN excluded.exhibition_hall ELSE books.exhibition_hall END,
                    highlights_json = CASE WHEN excluded.highlights_json != '[]' THEN excluded.highlights_json ELSE books.highlights_json END,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                slug, title, category, tags, total_pages, word_count,
                file_size_bytes, source_type, source_id, cover_image,
                summary, is_public, status,
                artifact_code, period_era, provenance,
                material_condition, curator_notes, exhibition_hall,
                highlights_json
            ))
            conn.commit()

        return self.get_book_by_slug(slug)

    def get_book_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM books WHERE slug = ? LIMIT 1;", (slug,)).fetchone()
            if row:
                return dict(row)
        return None

    def get_book_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        slug = slugify(title)
        return self.get_book_by_slug(slug)

    def find_duplicate(
        self,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        title: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if book already exists and is completed.
        Matches by source_id (e.g. gdrive folder id) or slug/title.
        """
        with self._get_connection() as conn:
            if source_id:
                row = conn.execute(
                    "SELECT * FROM books WHERE source_id = ? AND status = 'COMPLETED' LIMIT 1;",
                    (source_id,)
                ).fetchone()
                if row:
                    return dict(row)

            if title:
                slug = slugify(title)
                row = conn.execute(
                    "SELECT * FROM books WHERE (slug = ? OR title = ?) AND status = 'COMPLETED' LIMIT 1;",
                    (slug, title)
                ).fetchone()
                if row:
                    return dict(row)

        return None

    def list_books(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort: str = "newest",
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        with self._get_connection() as conn:
            where_clauses = ["is_public = 1"]
            params = []

            if category and category != "all":
                where_clauses.append("category = ?")
                params.append(category)

            if search:
                where_clauses.append("(title LIKE ? OR tags LIKE ? OR summary LIKE ?)")
                keyword = f"%{search.strip()}%"
                params.extend([keyword, keyword, keyword])

            where_sql = " WHERE " + " AND ".join(where_clauses)

            order_sql = " ORDER BY created_at DESC"
            if sort == "pages_desc":
                order_sql = " ORDER BY total_pages DESC"
            elif sort == "pages_asc":
                order_sql = " ORDER BY total_pages ASC"
            elif sort == "title_asc":
                order_sql = " ORDER BY title ASC"

            count_row = conn.execute(f"SELECT COUNT(*) as cnt FROM books {where_sql};", params).fetchone()
            total = count_row["cnt"] if count_row else 0

            params.extend([limit, offset])
            rows = conn.execute(
                f"SELECT * FROM books {where_sql} {order_sql} LIMIT ? OFFSET ?;",
                params
            ).fetchall()

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "books": [dict(r) for r in rows]
            }

    def get_categories_stats(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM books
                WHERE is_public = 1
                GROUP BY category
                ORDER BY count DESC;
            """).fetchall()
            return [dict(r) for r in rows]

    def update_metadata(
        self,
        slug: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        summary: Optional[str] = None,
        is_public: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        fields = []
        params = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if category is not None:
            fields.append("category = ?")
            params.append(category)
        if tags is not None:
            fields.append("tags = ?")
            params.append(tags)
        if summary is not None:
            fields.append("summary = ?")
            params.append(summary)
        if is_public is not None:
            fields.append("is_public = ?")
            params.append(is_public)

        if not fields:
            return self.get_book_by_slug(slug)

        fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(slug)

        with self._get_connection() as conn:
            conn.execute(f"UPDATE books SET {', '.join(fields)} WHERE slug = ?;", params)
            conn.commit()

        return self.get_book_by_slug(slug)

    def sync_all_books(self) -> List[Dict[str, Any]]:
        """
        Auto-scan data/output/ directory and sync all completed books with metadata,
        word counts, file sizes, and covers.
        """
        synced = []
        if not self.output_dir.is_dir():
            return synced

        book_names = set()
        for item in sorted(self.output_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                book_names.add(item.name)
            elif item.is_file() and item.suffix == ".txt" and not item.name.startswith("."):
                book_names.add(item.stem)

        for book_name in sorted(book_names):
            book_dir = self.output_dir / book_name
            text_file = self.output_dir / f"{book_name}.txt"

            total_pages = 0
            if book_dir.is_dir():
                total_pages = len([
                    f for f in book_dir.iterdir()
                    if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith(".")
                ])

            word_count = 0
            file_size = 0
            summary = ""
            if text_file.is_file():
                file_size = text_file.stat().st_size
                try:
                    content = text_file.read_text(encoding="utf-8", errors="ignore")
                    words = content.split()
                    word_count = len(words)
                    # Extract sample preview for summary
                    clean_lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("===")]
                    summary = " ".join(clean_lines[:5])[:300] + "..." if clean_lines else ""
                except Exception:
                    pass

            slug = slugify(book_name)
            cover_url = self.generate_cover_thumbnail(book_name, slug)

            book = self.upsert_book(
                title=book_name,
                slug=slug,
                total_pages=total_pages,
                word_count=word_count,
                file_size_bytes=file_size,
                cover_image=cover_url,
                summary=summary,
                status="COMPLETED"
            )
            synced.append(book)

        return synced

    def _natural_sort_key(self, s: Any) -> list:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

    def get_page_image_path(self, slug: str, page_num: int) -> Optional[Path]:
        """
        Resolve the original scanned/photo image for page `page_num` of book `slug`.
        If the original image is HEIC/HEIF or large raw format, converts and caches as JPEG.
        """
        book = self.get_book_by_slug(slug)
        if not book:
            return None

        book_input = self.input_dir / book["title"]
        if not book_input.is_dir():
            if self.input_dir.is_dir():
                candidates = [d for d in self.input_dir.iterdir() if d.is_dir() and slugify(d.name) == slug]
                book_input = candidates[0] if candidates else None
            else:
                book_input = None

        if book_input and book_input.is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}
            images = sorted(
                [f for f in book_input.iterdir() if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith(".")],
                key=lambda f: self._natural_sort_key(f.name)
            )

            if images and 1 <= page_num <= len(images):
                src_image = images[page_num - 1]
                ext = src_image.suffix.lower()

                # If it's a web-compatible image format already, return it directly
                if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                    return src_image

                # If it's HEIC or TIFF, convert and cache as JPEG
                try:
                    cached_file.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(src_image) as im:
                        im = im.convert("RGB")
                        im.save(cached_file, format="JPEG", quality=88, optimize=True)
                    return cached_file
                except Exception as e:
                    print(f"[BookRepository] Error converting {src_image.name} to cached JPEG: {e}")

        # Fallback for page 1 if cover exists
        if page_num == 1:
            cover = self.covers_dir / f"{slug}.jpg"
            if cover.is_file() and cover.stat().st_size > 0:
                return cover

        return None

    def get_artifact_dossier(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Get complete Museum Artifact Dossier: metadata + per-page data with
        aligned original scan images and clean OCR text.
        """
        book = self.get_book_by_slug(slug)
        if not book:
            return None

        book_dir = self.output_dir / book["title"]
        if not book_dir.is_dir():
            if self.output_dir.is_dir():
                candidates = [d for d in self.output_dir.iterdir() if d.is_dir() and slugify(d.name) == slug]
                book_dir = candidates[0] if candidates else None
            else:
                book_dir = None

        # Scan input images count for pairing
        book_input = self.input_dir / book["title"]
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}
        total_images = 0
        if book_input.is_dir():
            total_images = len([f for f in book_input.iterdir() if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith(".")])
        elif self.input_dir.is_dir():
            for d in self.input_dir.iterdir():
                if d.is_dir() and slugify(d.name) == slug:
                    total_images = len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith(".")])
                    break

        pages = []
        if book_dir and book_dir.is_dir():
            txt_files = sorted(
                [f for f in book_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith(".")],
                key=lambda f: self._natural_sort_key(f.name)
            )
            for idx, txt_file in enumerate(txt_files):
                page_num = idx + 1
                try:
                    content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    content = ""

                has_img = (page_num <= total_images) or (page_num == 1 and bool(book.get("cover_image")))
                pages.append({
                    "page_num": page_num,
                    "filename": txt_file.name,
                    "has_image": has_img,
                    "image_url": f"/api/museum/artifacts/{slug}/page-image/{page_num}" if has_img else "",
                    "text": content
                })

        # Fallback if no separate page files but aggregated text file exists
        if not pages:
            text_file = self.output_dir / f"{book['title']}.txt"
            if not text_file.is_file() and self.output_dir.is_dir():
                for f in self.output_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".txt" and slugify(f.stem) == slug:
                        text_file = f
                        break

            if text_file.is_file():
                try:
                    full_text = text_file.read_text(encoding="utf-8", errors="ignore")
                    # Split by === [Trang X/Y] ===
                    page_regex = re.compile(r"===\s*\[Trang\s+(\d+)/\d+\]\s+([^=\n\r]+)\s*===", re.IGNORECASE)
                    matches = list(page_regex.finditer(full_text))
                    if matches:
                        for idx, m in enumerate(matches):
                            p_num = int(m.group(1))
                            start = m.end()
                            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
                            p_text = full_text[start:end].strip()
                            has_img = (p_num <= total_images) or (p_num == 1 and bool(book.get("cover_image")))
                            pages.append({
                                "page_num": p_num,
                                "filename": m.group(2).strip(),
                                "has_image": has_img,
                                "image_url": f"/api/museum/artifacts/{slug}/page-image/{p_num}" if has_img else "",
                                "text": p_text
                            })
                    else:
                        has_img = bool(book.get("cover_image"))
                        pages.append({
                            "page_num": 1,
                            "filename": "full.txt",
                            "has_image": has_img,
                            "image_url": f"/api/museum/artifacts/{slug}/page-image/1" if has_img else "",
                            "text": full_text
                        })
                except Exception:
                    pass

        # Final guarantee: never return an empty pages list
        if not pages:
            summary = book.get("summary") or "Hiện vật đã được lưu trữ trong danh mục di sản số."
            has_img = bool(book.get("cover_image"))
            pages.append({
                "page_num": 1,
                "filename": f"{slug}_p1.txt",
                "has_image": has_img,
                "image_url": f"/api/museum/artifacts/{slug}/page-image/1" if has_img else "",
                "text": summary
            })

        return {
            "artifact": book,
            "total_pages": len(pages),
            "pages": pages,
            "curator_dossier": {
                "artifact_code": book.get("artifact_code") or f"HERITAGE-{book['id']:04d}",
                "period_era": book.get("period_era") or "Tư liệu Lịch sử & Văn hóa",
                "provenance": book.get("provenance") or "Kho tư liệu số hóa Book OCR Studio",
                "material_condition": book.get("material_condition") or "Bản lưu trữ số hóa độ phân giải cao",
                "curator_notes": book.get("curator_notes") or book.get("summary") or "Tư liệu quý được số hóa và phục chế văn bản bằng AI.",
                "exhibition_hall": book.get("exhibition_hall") or "Khu Trưng Bày Chung",
                "citations": {
                    "apa": f"{book['title']}. (2026). Lưu trữ số hóa tại Bảo Tàng Số Book OCR Studio. https://cuongblowurmind-book-ocr-studio.hf.space/museum/exhibit/{slug}",
                    "bibtex": f"@book{{{slug},\n  title = {{{book['title']}}},\n  year = {{2026}},\n  publisher = {{Bảo Tàng Số Book OCR Studio}},\n  url = {{https://cuongblowurmind-book-ocr-studio.hf.space/museum/exhibit/{slug}}}\n}}",
                    "tcvn": f"{book['title']}. Bản số hóa di sản, Bảo Tàng Số Book OCR Studio, 2026."
                }
            }
        }

    def get_museum_exhibits(self) -> Dict[str, Any]:
        """
        Get all curated exhibition halls with their artifacts and statistics.
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM books
                WHERE is_public = 1
                ORDER BY created_at DESC;
            """).fetchall()
            all_books = [dict(r) for r in rows]

        # Standard halls definition
        HALL_METADATA = {
            "Huyền tích & Tín ngưỡng Sông Hồng": {
                "id": "song-hong-heritage",
                "title": "Huyền tích & Tín ngưỡng Sông Hồng",
                "subtitle": "Ký ức phù sa, Tứ Bất Tử và tín ngưỡng ngàn năm châu thổ",
                "icon": "ph-waves",
                "color": "from-amber-600/30 to-red-900/30",
                "border": "border-amber-500/30"
            },
            "Địa phương chí & Ký ức Làng xã": {
                "id": "dia-phuong-chi",
                "title": "Địa phương chí & Ký ức Làng xã",
                "subtitle": "Gia phả, địa bạ, hương ước và di tích các vùng đất văn hiến",
                "icon": "ph-house-line",
                "color": "from-emerald-600/30 to-teal-900/30",
                "border": "border-emerald-500/30"
            },
            "Di sản Khoa học & Giáo dục": {
                "id": "khoa-hoc-giao-duc",
                "title": "Di sản Khoa học & Giáo dục",
                "subtitle": "Luận án tiến sĩ, chuyên khảo khoa học và tài liệu nghiên cứu ứng dụng",
                "icon": "ph-graduation-cap",
                "color": "from-blue-600/30 to-indigo-900/30",
                "border": "border-blue-500/30"
            },
            "Khu Trưng Bày Chung": {
                "id": "khu-trung-bay-chung",
                "title": "Bộ Sưu Tập Tổng Hợp",
                "subtitle": "Các ấn phẩm văn hóa và tư liệu đa dạng đã được giám định số",
                "icon": "ph-archive",
                "color": "from-purple-600/30 to-slate-900/30",
                "border": "border-purple-500/30"
            }
        }

        halls_dict = {}
        total_pages = 0
        total_words = 0

        for b in all_books:
            hall_name = b.get("exhibition_hall") or "Khu Trưng Bày Chung"
            if hall_name not in HALL_METADATA:
                hall_name = "Khu Trưng Bày Chung"

            if hall_name not in halls_dict:
                halls_dict[hall_name] = {
                    **HALL_METADATA[hall_name],
                    "name": hall_name,
                    "artifacts": []
                }

            halls_dict[hall_name]["artifacts"].append(b)
            total_pages += (b.get("total_pages") or 0)
            total_words += (b.get("word_count") or 0)

        # Ensure all standard halls appear even if empty
        for hall_name, meta in HALL_METADATA.items():
            if hall_name not in halls_dict and hall_name != "Khu Trưng Bày Chung":
                halls_dict[hall_name] = {
                    **meta,
                    "name": hall_name,
                    "artifacts": []
                }

        halls_list = list(halls_dict.values())

        return {
            "stats": {
                "total_artifacts": len(all_books),
                "total_pages": total_pages,
                "total_words": total_words,
                "total_halls": len(halls_list)
            },
            "halls": halls_list
        }

    def seed_museum_metadata(self) -> None:
        """
        Seed museum curation metadata for the 4 primary books if empty.
        """
        SEEDS = [
            {
                "slug": "tho-chu-dong-tu",
                "artifact_code": "MUSEUM-HERITAGE-001",
                "period_era": "Thế kỷ XX • Khảo cứu văn hóa dân gian",
                "provenance": "Sưu tập tư liệu điền dã Đền Đa Hòa - Dạ Trạch",
                "material_condition": "Ảnh chụp tài liệu gốc tại thực địa, có thủ bút và triện",
                "curator_notes": "Công trình ghi chép toàn diện về một trong 'Tứ bất tử' của tín ngưỡng dân gian Việt Nam, bảo tồn các nghi thức rước nước, tế lễ cổ truyền.",
                "exhibition_hall": "Huyền tích & Tín ngưỡng Sông Hồng"
            },
            {
                "slug": "dat-yen-my-hung-yen",
                "artifact_code": "MUSEUM-HERITAGE-002",
                "period_era": "Thế kỷ XIX - XX • Địa phương chí",
                "provenance": "Tài liệu địa chí lưu trữ huyện Yên Mỹ, tỉnh Hưng Yên",
                "material_condition": "Tư liệu in nguyên bản, bản chụp độ phân giải cao",
                "curator_notes": "Tập đại thành địa chí ghi nhận lịch sử hình thành các tổng, làng, dòng họ, phong tục tập quán và di tích lịch sử vùng đất Hưng Yên văn hiến.",
                "exhibition_hall": "Địa phương chí & Ký ức Làng xã"
            },
            {
                "slug": "pp-nc-khoa-hoc-du-lich-giao-trinh",
                "artifact_code": "MUSEUM-HERITAGE-003",
                "period_era": "Hiện đại • Thập niên 2000",
                "provenance": "Thư viện Giáo trình Đại học Quốc gia",
                "material_condition": "Giáo trình chuyên khảo hoàn chỉnh, lưu trữ số nguyên bản",
                "curator_notes": "Hệ thống phương pháp luận nghiên cứu du lịch học ứng dụng, làm sáng tỏ các giá trị di sản và kinh tế văn hóa.",
                "exhibition_hall": "Di sản Khoa học & Giáo dục"
            },
            {
                "slug": "luan-an-song-hong-1",
                "artifact_code": "MUSEUM-HERITAGE-004",
                "period_era": "Thế kỷ XX • Nghiên cứu địa lý & thủy văn",
                "provenance": "Viện Nghiên cứu Địa lý & Môi trường",
                "material_condition": "Bản luận án đánh máy cổ điển kèm sơ đồ đo đạc",
                "curator_notes": "Công trình nghiên cứu chuyên sâu về bồi tích, hệ sinh thái và tác động của sông Hồng đến đời sống cư dân đồng bằng Bắc Bộ.",
                "exhibition_hall": "Huyền tích & Tín ngưỡng Sông Hồng"
            }
        ]

        with self._get_connection() as conn:
            for s in SEEDS:
                conn.execute("""
                    UPDATE books SET
                        artifact_code = CASE WHEN artifact_code = '' THEN ? ELSE artifact_code END,
                        period_era = CASE WHEN period_era = '' THEN ? ELSE period_era END,
                        provenance = CASE WHEN provenance = '' THEN ? ELSE provenance END,
                        material_condition = CASE WHEN material_condition = '' THEN ? ELSE material_condition END,
                        curator_notes = CASE WHEN curator_notes = '' THEN ? ELSE curator_notes END,
                        exhibition_hall = CASE WHEN exhibition_hall = 'Khu Trưng Bày Chung' THEN ? ELSE exhibition_hall END
                    WHERE slug = ?;
                """, (
                    s["artifact_code"], s["period_era"], s["provenance"],
                    s["material_condition"], s["curator_notes"], s["exhibition_hall"],
                    s["slug"]
                ))
            conn.commit()
