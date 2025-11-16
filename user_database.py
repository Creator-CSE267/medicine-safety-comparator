# user_database.py
import sqlite3
from passlib.context import CryptContext
from pathlib import Path

DB_PATH = "users.db"
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def init_user_db():
    """
    Create the users DB and insert default admin/pharmacist users if they don't exist.
    Default credentials (change after first login):
      - admin / admin123
      - pharmacist / pharma123
    """
    db_file = Path(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Default users (only inserted if username not present)
    defaults = [
        ("admin", "Administrator", pwd_context.hash("admin123"), "admin"),
        ("pharmacist", "Pharmacist User", pwd_context.hash("pharma123"), "pharmacist"),
    ]

    for username, name, pw_hash, role in defaults:
        try:
            cur.execute(
                "INSERT INTO users (username, name, password_hash, role) VALUES (?, ?, ?, ?)",
                (username, name, pw_hash, role)
            )
        except sqlite3.IntegrityError:
            # user already exists
            pass

    conn.commit()
    conn.close()


def get_user(username):
    """
    Return user row as tuple: (username, name, password_hash, role) or None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username, name, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row  # None or tuple


def update_password(username, new_plain_password):
    """
    Hashes the new password and updates the user's password_hash.
    Returns True on success, False if user not found.
    """
    new_hash = pwd_context.hash(new_plain_password)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def verify_password(plain_password, stored_hash):
    """
    Verify plain password against stored hash. Returns True/False.
    """
    try:
        return pwd_context.verify(plain_password, stored_hash)
    except Exception:
        return False
