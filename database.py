"""
Thin SQLite wrapper for logging count events locally.

Schema
sessions   : one row per packing session (start time, end time, final count)
events     : one row per state machine event (ADDED / REMOVED / HIDDEN)

Design rules
- All writes go through a single sqlite3 connection owned by the calling
  thread (Thread 3). SQLite in WAL mode handles concurrent reads from
  other threads (e.g. a /history HTTP endpoint) safely.
- Exposes only four methods so the rest of the codebase never writes raw SQL.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CountDatabase:
    """
    Manages a local SQLite database for session and event logging.
    Parameters
    db_path : Path to the .db file. Parent directory must exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # WAL mode: readers don't block writers and vice versa.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL

        self._create_tables()
        self._session_id: Optional[int] = None
        logger.info("CountDatabase ready at %s", self._path)

    # Session management

    def start_session(self) -> int:
        """
        Open a new packing session. Returns the new session_id.
        Call this when the operator clicks "New Carton".
        """
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (time.time(),),
        )
        self._conn.commit()
        self._session_id = cur.lastrowid
        logger.info("DB session %d started", self._session_id)
        return self._session_id  # type: ignore[return-value]

    def end_session(self, final_count: int) -> None:
        """
        Close the current session with a final count and wall clock timestamp.
        """
        if self._session_id is None:
            return
        self._conn.execute(
            "UPDATE sessions SET ended_at=?, final_count=? WHERE id=?",
            (time.time(), final_count, self._session_id),
        )
        self._conn.commit()
        logger.info(
            "DB session %d ended | final_count=%d", self._session_id, final_count
        )

    # Event logging

    def log_event(
        self,
        event_type: str,   # "ADDED" | "REMOVED" | "HIDDEN"
        track_id:   int,
        count_after: int,
    ) -> None:
        """
        Record a single state machine event.
        Silently no ops if no session is open (avoids crashing on startup).
        """
        if self._session_id is None:
            return
        self._conn.execute(
            """INSERT INTO events
                   (session_id, event_type, track_id, count_after, ts)
               VALUES (?, ?, ?, ?, ?)""",
            (self._session_id, event_type, track_id, count_after, time.time()),
        )
        # No commit here, we batch commit in flush() for throughput.

    def flush(self) -> None:
        """
        Commit any pending event rows to disk.
        Call this from Thread 3 at the end of each frame (cheap; WAL mode).
        """
        self._conn.commit()

    # Query helpers (used by the /history REST endpoint in main.py)

    def get_session_summary(self) -> list[dict]:
        """Return all sessions as a list of dicts, newest first."""
        cur = self._conn.execute(
            """SELECT id, started_at, ended_at, final_count
               FROM sessions ORDER BY id DESC LIMIT 50"""
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_events_for_session(self, session_id: int) -> list[dict]:
        """Return all events for a specific session, oldest first."""
        cur = self._conn.execute(
            """SELECT event_type, track_id, count_after, ts
               FROM events WHERE session_id=? ORDER BY ts ASC""",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # Private

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at   REAL NOT NULL,
                ended_at     REAL,
                final_count  INTEGER
            );

            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL REFERENCES sessions(id),
                event_type   TEXT    NOT NULL,
                track_id     INTEGER NOT NULL,
                count_after  INTEGER NOT NULL,
                ts           REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id);
        """)
        self._conn.commit()