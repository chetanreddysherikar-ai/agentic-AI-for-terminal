"""Conversation memory backed by a local SQLite checkpoint database."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


class SessionStore:
    """Persistent, thread-based conversation memory.

    Each session name maps to a `thread_id`, so history survives across CLI
    restarts. Use `new_session()` to start a fresh thread.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._saver = SqliteSaver(self._conn)
        self._counter = 0

    @property
    def saver(self) -> SqliteSaver:
        return self._saver

    def config(self, thread_id: str) -> dict:
        """RunnableConfig for a given thread."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 60,
        }

    def new_session(self) -> str:
        """Generate a fresh, unused session name."""
        self._counter += 1
        return f"session-{self._counter}"

    def threads(self) -> list[str]:
        """List known session ids (most recent first, best effort)."""
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
            ).fetchall()
        except sqlite3.Error:
            return []
        return [row[0] for row in rows]

    def clear(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        self._conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
