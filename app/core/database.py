import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trustlens.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_HISTORY_COLUMNS = ("id", "query", "provider", "trust_score", "geo_score", "timestamp")


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            provider TEXT,
            trust_score REAL,
            geo_score REAL,
            timestamp TEXT
        )
        """
    )
    # Backward-compatible migration for older DBs without geo_score.
    try:
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(history)").fetchall()]
        if "geo_score" not in cols:
            cursor.execute("ALTER TABLE history ADD COLUMN geo_score REAL")
    except Exception:
        # Avoid blocking app startup on a failed migration attempt.
        pass
    conn.commit()
    conn.close()

