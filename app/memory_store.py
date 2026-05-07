import sqlite3
from pathlib import Path
from threading import Lock
from typing import List

from app.schemas import Message


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 8) -> List[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, id
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) recent
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [Message(role=row[0], content=row[1]) for row in rows]

    def append_messages(self, session_id: str, messages: List[Message]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO conversation_messages (session_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    [(session_id, m.role, m.content) for m in messages],
                )
                conn.commit()
