import sqlite3
from passlib.context import CryptContext
from pathlib import Path

DB_PATH = "users.db"
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def init_user_db():
    """
    Creates DB and inserts 2 admin + 6 pharmacists.
    Runs only once (will NOT overwrite existing users).
    """

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # ---- DEFAULT USERS ----
    default_admins = [
        ("admin1", "Admin One", pwd_context.hash("admin123"), "admin"),
        ("admin2", "Admin Two", pwd_context.hash("admin123"), "admin"),
    ]

    default_pharmacists = [
        ("pharma1", "Pharmacist 01", pwd_context.hash("pharma123"), "pharmacist"),
        ("pharma2", "Pharmacist 02", pwd_context.hash("pharma123"), "pharmacist"),
        ("pharma3", "Pharmacist 03", pwd_context.hash("pharma123"), "pharmacist"),
        ("pharma4", "Pharmacist 04", pwd_context.hash("pharma123"), "pharmacist"),
        ("pharma5", "Pharmacist 05", pwd_context.hash("pharma123"), "pharmacist"),
        ("pharma6", "Pharmacist 06", pwd_context.hash("pharma123"), "pharmacist"),
    ]

    # Insert without overwriting
    for user in default_admins + default_pharmacists:
        try:
            cur.execute(
                "INSERT INTO users (username, name, password_hash, role) VALUES (?, ?, ?, ?)",
                (user[0], user[1], user[2], user[3])
            )
        except sqlite3.IntegrityError:
            pass  # already exists

    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username, name, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def update_password(username, new_pass):
    new_hash = pwd_context.hash(new_pass)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    conn.commit()
    ok = cur.rowcount
    conn.close()
    return ok > 0


def verify_password(plain, stored_hash):
    try:
        return pwd_context.verify(plain, stored_hash)
    except Exception:
        return False
