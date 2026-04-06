"""SQLite-backed checkpoint system for tracking scraping progress."""
import sqlite3
import threading
import time
from pathlib import Path


class Checkpoint:
    """
    Persistent checkpoint tracker using SQLite.

    Table schema:
        progress (step TEXT, key TEXT, status TEXT, ts REAL, PRIMARY KEY (step, key))

    Thread-safe: uses threading.Lock on all write operations.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS progress (
                    step TEXT NOT NULL,
                    key  TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ts   REAL NOT NULL,
                    PRIMARY KEY (step, key)
                )
                """
            )
            conn.commit()

    def is_done(self, step: str, key: str) -> bool:
        """Return True if (step, key) is marked as 'done'."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM progress WHERE step = ? AND key = ?",
                (step, key),
            ).fetchone()
        return row is not None and row["status"] == "done"

    def mark_done(self, step: str, key: str) -> None:
        """Mark (step, key) as done."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO progress (step, key, status, ts)
                    VALUES (?, ?, 'done', ?)
                    ON CONFLICT(step, key) DO UPDATE SET status='done', ts=excluded.ts
                    """,
                    (step, key, time.time()),
                )
                conn.commit()

    def mark_error(self, step: str, key: str) -> None:
        """Mark (step, key) as error."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO progress (step, key, status, ts)
                    VALUES (?, ?, 'error', ?)
                    ON CONFLICT(step, key) DO UPDATE SET status='error', ts=excluded.ts
                    """,
                    (step, key, time.time()),
                )
                conn.commit()

    def pending_count(self, step: str) -> int:
        """Return count of rows for this step that are not 'done'."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM progress WHERE step = ? AND status != 'done'",
                (step,),
            ).fetchone()
        return row["cnt"] if row else 0

    def done_count(self, step: str) -> int:
        """Return count of rows for this step that are 'done'."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM progress WHERE step = ? AND status = 'done'",
                (step,),
            ).fetchone()
        return row["cnt"] if row else 0
