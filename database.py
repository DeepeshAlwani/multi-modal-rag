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
        # Users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Rate limiting table
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
        
        # User sessions
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
        
        # Repo indexing jobs
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
        
        conn.commit()

def create_user(email: str, password: str) -> bool:
    """Create new user"""
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
    """Verify user credentials"""
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        
        if user and bcrypt.checkpw(password.encode(), user['password_hash']):
            return dict(user)
    return None

def check_rate_limit(user_id: int, ip_address: str, limit: int = 100, window_seconds: int = 3600) -> bool:
    """Check if user has exceeded rate limit"""
    with get_db() as conn:
        # Clean old entries
        conn.execute(
            "DELETE FROM rate_limits WHERE window_start < datetime('now', ?)",
            (f'-{window_seconds} seconds',)
        )
        
        # Get current count
        result = conn.execute(
            """SELECT SUM(request_count) as total FROM rate_limits 
               WHERE (user_id = ? OR ip_address = ?) 
               AND window_start > datetime('now', ?)""",
            (user_id, ip_address, f'-{window_seconds} seconds')
        ).fetchone()
        
        total = result['total'] or 0
        
        if total >= limit:
            return False
        
        # Increment counter
        conn.execute(
            """INSERT INTO rate_limits (user_id, ip_address, request_count) 
               VALUES (?, ?, 1)""",
            (user_id, ip_address)
        )
        conn.commit()
        return True

def create_session(user_id: int) -> str:
    """Create a new session token"""
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
    """Verify session token"""
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
            return dict(user)
    return None

def add_repo_job(user_id: int, repo_url: str, repo_path: str):
    """Add repo indexing job"""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO repos (user_id, repo_url, repo_path, status) VALUES (?, ?, ?, 'pending')",
            (user_id, repo_url, repo_path)
        )
        conn.commit()

init_db()