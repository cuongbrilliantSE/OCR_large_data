import sqlite3
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


class StateTracker:
    def __init__(self, db_path: str | Path = "data/tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency read/write operations
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    confidence REAL DEFAULT NULL,
                    error_msg TEXT DEFAULT NULL,
                    processing_time_ms REAL DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON tasks(file_path);")

    def register_files(self, file_entries: List[Tuple[str, int]]) -> int:
        """
        Batch register scanned files into the database.
        Uses INSERT OR IGNORE to ensure idempotency.
        Returns number of newly registered files.
        """
        if not file_entries:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO tasks (file_path, file_size, status)
                VALUES (?, ?, 'PENDING');
            """, file_entries)
            return cursor.rowcount

    def reset_hanging_tasks(self, book_name: Optional[str] = None) -> int:
        """
        Reset any task stuck in 'IN_PROGRESS' back to 'PENDING'.
        Call this when starting/resuming after an unexpected shutdown.
        """
        query = "UPDATE tasks SET status = 'PENDING', updated_at = CURRENT_TIMESTAMP WHERE status = 'IN_PROGRESS'"
        params = []
        if book_name:
            query += " AND (file_path LIKE ? OR file_path LIKE ?)"
            params.extend([f"{book_name}/%", f"{book_name}\\%"])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def retry_failed_tasks(self, book_name: Optional[str] = None) -> int:
        """Reset all 'FAILED' tasks back to 'PENDING'."""
        query = "UPDATE tasks SET status = 'PENDING', error_msg = NULL, updated_at = CURRENT_TIMESTAMP WHERE status = 'FAILED'"
        params = []
        if book_name:
            query += " AND (file_path LIKE ? OR file_path LIKE ?)"
            params.extend([f"{book_name}/%", f"{book_name}\\%"])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def get_pending_tasks(
        self,
        limit: Optional[int] = None,
        book_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch pending tasks, optionally filtered by book_name."""
        query = "SELECT file_path, file_size FROM tasks WHERE status = 'PENDING'"
        params = []
        if book_name:
            query += " AND (file_path LIKE ? OR file_path LIKE ?)"
            params.extend([f"{book_name}/%", f"{book_name}\\%"])
        query += " ORDER BY id ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def mark_in_progress(self, file_paths: List[str]) -> None:
        """Mark a batch of tasks as IN_PROGRESS."""
        if not file_paths:
            return
        with self._get_connection() as conn:
            conn.executemany("""
                UPDATE tasks
                SET status = 'IN_PROGRESS', updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?;
            """, [(fp,) for fp in file_paths])

    def batch_update_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Batch update task results after worker execution.
        Expected result format:
        {
            "file_path": str,
            "success": bool,
            "confidence": float,
            "error_msg": Optional[str],
            "processing_time_ms": float
        }
        """
        if not results:
            return

        with self._get_connection() as conn:
            for item in results:
                status = "DONE" if item.get("success") else "FAILED"
                conn.execute("""
                    UPDATE tasks
                    SET status = ?,
                        confidence = ?,
                        error_msg = ?,
                        processing_time_ms = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?;
                """, (
                    status,
                    item.get("confidence"),
                    item.get("error_msg"),
                    item.get("processing_time_ms"),
                    item["file_path"]
                ))

    def get_statistics(self, book_name: Optional[str] = None) -> Dict[str, Any]:
        """Return counts by status and overall progress, optionally filtered by book_name."""
        query = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status = 'IN_PROGRESS' THEN 1 ELSE 0 END) AS in_progress,
                SUM(CASE WHEN status = 'DONE' THEN 1 ELSE 0 END) AS done,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed,
                AVG(CASE WHEN status = 'DONE' THEN processing_time_ms ELSE NULL END) AS avg_time_ms,
                AVG(CASE WHEN status = 'DONE' THEN confidence ELSE NULL END) AS avg_confidence
            FROM tasks
        """
        params = []
        if book_name:
            query += " WHERE (file_path LIKE ? OR file_path LIKE ?)"
            params.extend([f"{book_name}/%", f"{book_name}\\%"])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            total = row["total"] or 0
            done = row["done"] or 0
            pct = (done / total * 100.0) if total > 0 else 0.0

            return {
                "total": total,
                "pending": row["pending"] or 0,
                "in_progress": row["in_progress"] or 0,
                "done": done,
                "failed": row["failed"] or 0,
                "percent_complete": round(pct, 2),
                "avg_time_ms": round(row["avg_time_ms"] or 0.0, 2),
                "avg_confidence": round(row["avg_confidence"] or 0.0, 4)
            }

    def get_book_statistics(self) -> List[Dict[str, Any]]:
        """
        Return status counts aggregated per book folder (the top-level directory of file_path).
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    CASE 
                        WHEN instr(file_path, '/') > 0 THEN substr(file_path, 1, instr(file_path, '/') - 1)
                        WHEN instr(file_path, '\\') > 0 THEN substr(file_path, 1, instr(file_path, '\\') - 1)
                        ELSE '[root]'
                    END AS book_name,
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN status = 'IN_PROGRESS' THEN 1 ELSE 0 END) AS in_progress,
                    SUM(CASE WHEN status = 'DONE' THEN 1 ELSE 0 END) AS done,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed
                FROM tasks
                GROUP BY book_name
                HAVING NOT book_name LIKE '.%'
                ORDER BY book_name ASC;
            """)
            books = []
            for row in cursor.fetchall():
                total = row["total"] or 0
                done = row["done"] or 0
                pct = round((done / total * 100.0), 1) if total > 0 else 0.0
                books.append({
                    "book_name": row["book_name"],
                    "total": total,
                    "pending": row["pending"] or 0,
                    "in_progress": row["in_progress"] or 0,
                    "done": done,
                    "failed": row["failed"] or 0,
                    "percent": pct
                })
            return books

