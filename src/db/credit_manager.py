import sqlite3
import uuid
import secrets
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


class CreditManager:
    """
    Manages user sessions, credit balances (free trial + paid credits),
    vouchers (license keys), and custom Gemini API keys (BYOK).
    """

    def __init__(self, db_path: str | Path = "data/tracker.db", default_free_pages: int = 10):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_free_pages = default_free_pages
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
                CREATE TABLE IF NOT EXISTS users (
                    user_token TEXT PRIMARY KEY,
                    free_pages_left INTEGER NOT NULL DEFAULT 10,
                    paid_credits INTEGER NOT NULL DEFAULT 0,
                    custom_api_key TEXT DEFAULT NULL,
                    role TEXT NOT NULL DEFAULT 'guest',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS vouchers (
                    code TEXT PRIMARY KEY,
                    credits INTEGER NOT NULL,
                    is_used BOOLEAN NOT NULL DEFAULT 0,
                    used_by TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    used_at TIMESTAMP DEFAULT NULL,
                    FOREIGN KEY (used_by) REFERENCES users(user_token)
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_token TEXT NOT NULL,
                    book_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    credit_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_vouchers_code ON vouchers(code);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_logs_user ON credit_logs(user_token);")

    def get_or_create_user(self, user_token: Optional[str] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            if user_token:
                cursor = conn.execute(
                    "SELECT * FROM users WHERE user_token = ?",
                    (user_token,)
                )
                row = cursor.fetchone()
                if row:
                    conn.execute(
                        "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_token = ?",
                        (user_token,)
                    )
                    return dict(row)

            new_token = user_token or f"user_{uuid.uuid4().hex[:16]}"
            conn.execute("""
                INSERT INTO users (user_token, free_pages_left, paid_credits, role)
                VALUES (?, ?, 0, 'guest')
                ON CONFLICT(user_token) DO UPDATE SET last_active = CURRENT_TIMESTAMP;
            """, (new_token, self.default_free_pages))

            cursor = conn.execute("SELECT * FROM users WHERE user_token = ?", (new_token,))
            return dict(cursor.fetchone())

    def get_user_balance(self, user_token: str) -> Dict[str, Any]:
        user = self.get_or_create_user(user_token)
        has_byok = bool(user.get("custom_api_key") and user["custom_api_key"].strip())
        free_left = max(0, user.get("free_pages_left", 0))
        paid_left = max(0, user.get("paid_credits", 0))

        return {
            "user_token": user["user_token"],
            "free_pages_left": free_left,
            "paid_credits": paid_left,
            "total_available": (free_left + paid_left) if not has_byok else 999999,
            "has_byok": has_byok,
            "masked_key": f"{user['custom_api_key'][:4]}...{user['custom_api_key'][-4:]}" if has_byok else None,
            "role": user.get("role", "guest")
        }

    def has_sufficient_credits(self, user_token: str) -> bool:
        user = self.get_or_create_user(user_token)
        if user.get("custom_api_key") and user["custom_api_key"].strip():
            return True
        return (user.get("free_pages_left", 0) + user.get("paid_credits", 0)) > 0

    def deduct_page_credit(self, user_token: str, book_name: str, file_path: str) -> Tuple[bool, str]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE user_token = ?", (user_token,))
            user = cursor.fetchone()
            if not user:
                return False, "user_not_found"

            if user["custom_api_key"] and user["custom_api_key"].strip():
                conn.execute("""
                    INSERT INTO credit_logs (user_token, book_name, file_path, credit_type)
                    VALUES (?, ?, ?, 'byok');
                """, (user_token, book_name, file_path))
                return True, "byok"

            if user["free_pages_left"] > 0:
                conn.execute("""
                    UPDATE users
                    SET free_pages_left = free_pages_left - 1,
                        last_active = CURRENT_TIMESTAMP
                    WHERE user_token = ?;
                """, (user_token,))
                conn.execute("""
                    INSERT INTO credit_logs (user_token, book_name, file_path, credit_type)
                    VALUES (?, ?, ?, 'free');
                """, (user_token, book_name, file_path))
                return True, "free"

            if user["paid_credits"] > 0:
                conn.execute("""
                    UPDATE users
                    SET paid_credits = paid_credits - 1,
                        last_active = CURRENT_TIMESTAMP
                    WHERE user_token = ?;
                """, (user_token,))
                conn.execute("""
                    INSERT INTO credit_logs (user_token, book_name, file_path, credit_type)
                    VALUES (?, ?, ?, 'paid');
                """, (user_token, book_name, file_path))
                return True, "paid"

            return False, "out_of_credits"

    def set_custom_api_key(self, user_token: str, custom_key: Optional[str]) -> bool:
        clean_key = custom_key.strip() if custom_key else None
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE users
                SET custom_api_key = ?, last_active = CURRENT_TIMESTAMP
                WHERE user_token = ?;
            """, (clean_key, user_token))
            return cursor.rowcount > 0

    def create_vouchers(self, credits: int, count: int = 1, prefix: str = "OCR") -> List[str]:
        created_codes = []
        with self._get_connection() as conn:
            for _ in range(count):
                random_part = secrets.token_hex(4).upper()
                code = f"{prefix}-{credits}P-{random_part}"
                conn.execute("""
                    INSERT INTO vouchers (code, credits)
                    VALUES (?, ?);
                """, (code, credits))
                created_codes.append(code)
        return created_codes

    def redeem_voucher(self, user_token: str, code: str) -> Tuple[bool, str, int]:
        clean_code = code.strip().upper()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM vouchers WHERE code = ?", (clean_code,))
            voucher = cursor.fetchone()
            if not voucher:
                return False, "Mã kích hoạt không tồn tại.", 0

            if voucher["is_used"]:
                return False, "Mã kích hoạt này đã được sử dụng trước đó.", 0

            credits = voucher["credits"]
            conn.execute("""
                UPDATE vouchers
                SET is_used = 1, used_by = ?, used_at = CURRENT_TIMESTAMP
                WHERE code = ?;
            """, (user_token, clean_code))

            conn.execute("""
                UPDATE users
                SET paid_credits = paid_credits + ?, last_active = CURRENT_TIMESTAMP
                WHERE user_token = ?;
            """, (credits, user_token))

            return True, f"Kích hoạt thành công! Đã cộng thêm {credits} trang sách vào tài khoản.", credits

    def list_vouchers(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT code, credits, is_used, used_by, created_at, used_at
                FROM vouchers
                ORDER BY created_at DESC
                LIMIT ?;
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def list_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT user_token, free_pages_left, paid_credits,
                       CASE WHEN custom_api_key IS NOT NULL AND custom_api_key != '' THEN 1 ELSE 0 END AS has_byok,
                       role, created_at, last_active
                FROM users
                ORDER BY last_active DESC
                LIMIT ?;
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
