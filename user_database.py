# user_database.py
import sqlite3
from passlib.context import CryptContext

DB_PATH = "users.db"
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def init_user_db():
    """Create users table and insert default admin/pharmacist accounts."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            password_hash TEXT,
            role TEXT
        )
    """)

    # Insert default users if they do not exist
    users = [
        ("admin", "Administrator", pwd_context.hash("admin123"), "admin"),
        ("pharmacist", "Pharmacist User", pwd_context.hash("pharma123"), "pharmacist")
    ]

    for u in users:
        try:
            cur.execute("INSERT INTO users (username, name, password_hash, role) VALUES (?, ?, ?, ?)", u)
        except:
            pass  # Already exists

    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username, name, password_hash, role FROM users WHERE username=?", (username,))
    user = cur.fetchone()
    conn.close()
    return user


def update_password(username, new_hash):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash=? WHERE username=?", (new_hash, username))
    conn.commit()
    conn.close()
