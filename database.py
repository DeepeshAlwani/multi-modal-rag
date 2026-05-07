import sqlite3
from datetime import datetime, timedelta
import bcrypt
import json
from contextlib import contextmanager

DATABASE_PATH = "users.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ip_address TEXT,
                request_count INTEGER DEFAULT 1,
                window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Legacy repo jobs table (kept for audit trail)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                repo_url TEXT,
                repo_path TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # NEW: active repo per user — survives server restarts
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_active_repo (
                user_id INTEGER PRIMARY KEY,
                repo_url TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                repo_hash TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(email: str, password: str) -> bool:
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, password_hash)
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False


def verify_user(email: str, password: str) -> dict | None:
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if user and bcrypt.checkpw(password.encode(), user['password_hash']):
            return dict(user)
    return None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(user_id: int) -> str:
    import secrets
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=7)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        conn.commit()
    return token


def verify_session(token: str) -> dict | None:
    """Return user dict if token is valid and not expired, else None."""
    if not token:
        return None
    with get_db() as conn:
        session = conn.execute(
            "SELECT user_id FROM sessions WHERE token = ? AND expires_at > CURRENT_TIMESTAMP",
            (token,)
        ).fetchone()
        if session:
            user = conn.execute(
                "SELECT id, email FROM users WHERE id = ?",
                (session['user_id'],)
            ).fetchone()
            return dict(user) if user else None
    return None


def delete_session(token: str) -> None:
    """Invalidate a session token on explicit logout."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit(user_id: int | None, ip_address: str, limit: int = 100, window_seconds: int = 3600) -> bool:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM rate_limits WHERE window_start < datetime('now', ?)",
            (f'-{window_seconds} seconds',)
        )

        conditions = []
        params = []

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        conditions.append("ip_address = ?")
        params.append(ip_address)
        params.append(f'-{window_seconds} seconds')

        result = conn.execute(
            f"""SELECT SUM(request_count) as total FROM rate_limits
               WHERE ({' OR '.join(conditions)})
               AND window_start > datetime('now', ?)""",
            params
        ).fetchone()

        total = result['total'] or 0
        if total >= limit:
            return False

        conn.execute(
            "INSERT INTO rate_limits (user_id, ip_address, request_count) VALUES (?, ?, 1)",
            (user_id, ip_address)
        )
        conn.commit()
        return True


# ---------------------------------------------------------------------------
# Active repo per user (persistent, replaces in-memory dict)
# ---------------------------------------------------------------------------

def upsert_user_repo(user_id: int, repo_url: str, repo_path: str, repo_hash: str) -> None:
    """Insert or replace the active repo record for a user."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO user_active_repo (user_id, repo_url, repo_path, repo_hash, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   repo_url   = excluded.repo_url,
                   repo_path  = excluded.repo_path,
                   repo_hash  = excluded.repo_hash,
                   updated_at = CURRENT_TIMESTAMP""",
            (user_id, repo_url, repo_path, repo_hash)
        )
        conn.commit()


def get_user_repo(user_id: int) -> dict | None:
    """Return the active repo info for a user, or None if not set."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT repo_url, repo_path, repo_hash FROM user_active_repo WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        return dict(row) if row else None


def clear_user_repo(user_id: int) -> None:
    """Remove the active repo record (e.g. when user clicks 'Change Repository')."""
    with get_db() as conn:
        conn.execute("DELETE FROM user_active_repo WHERE user_id = ?", (user_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Legacy job log
# ---------------------------------------------------------------------------

def add_repo_job(user_id: int, repo_url: str, repo_path: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO repos (user_id, repo_url, repo_path, status) VALUES (?, ?, ?, 'completed')",
            (user_id, repo_url, repo_path)
        )
        conn.commit()


init_db()