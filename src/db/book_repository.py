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
        covers_dir: str | Path = "data/covers"
    ):
        self.db_path = Path(db_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.covers_dir = Path(covers_dir)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.covers_dir.mkdir(parents=True, exist_ok=True)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_slug ON books(slug);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_category ON books(category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_books_source ON books(source_type, source_id);")

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
        status: str = "COMPLETED"
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
                    summary, is_public, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(slug) DO UPDATE SET
                    title = excluded.title,
                    total_pages = CASE WHEN excluded.total_pages > 0 THEN excluded.total_pages ELSE books.total_pages END,
                    word_count = CASE WHEN excluded.word_count > 0 THEN excluded.word_count ELSE books.word_count END,
                    file_size_bytes = CASE WHEN excluded.file_size_bytes > 0 THEN excluded.file_size_bytes ELSE books.file_size_bytes END,
                    cover_image = CASE WHEN excluded.cover_image != '' THEN excluded.cover_image ELSE books.cover_image END,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                slug, title, category, tags, total_pages, word_count,
                file_size_bytes, source_type, source_id, cover_image,
                summary, is_public, status
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
