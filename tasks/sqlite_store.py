from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, Optional

from config.settings import get_settings

from .models import (
    INTERRUPTED_ON_RESTART_MESSAGE,
    TaskRecord,
    TaskStatus,
    normalize_task_status,
)


DB_FILE = get_settings().task_db_file


class SQLiteTaskStore:
    def __init__(
        self,
        db_file: str = DB_FILE,
        *,
        timeout: float = 5.0,
        retry_count: int = 5,
        retry_delay: float = 0.2,
    ):
        self.db_file = db_file
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_file, timeout=self.timeout)

    def init_db(self) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS tasks
                         (id TEXT PRIMARY KEY,
                          status TEXT,
                          result TEXT,
                          message TEXT,
                          cost_time REAL DEFAULT 0,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
            )

            try:
                c.execute("ALTER TABLE tasks ADD COLUMN cost_time REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                c.execute("ALTER TABLE tasks ADD COLUMN created_at TIMESTAMP DEFAULT '1970-01-01 00:00:00'")
            except sqlite3.OperationalError:
                pass

            c.execute(
                "UPDATE tasks SET status=?, message=? WHERE status IN (?, ?)",
                (
                    TaskStatus.FAILED.value,
                    INTERRUPTED_ON_RESTART_MESSAGE,
                    TaskStatus.PENDING.value,
                    TaskStatus.PROCESSING.value,
                ),
            )
            c.execute("DELETE FROM tasks WHERE created_at <= datetime('now', '-3 days')")

    def create_task(self, task_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, status) VALUES (?, ?)",
                (task_id, TaskStatus.PENDING.value),
            )

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | str,
        result: Optional[Any] = None,
        message: Optional[str] = None,
        cost_time: float = 0,
    ) -> None:
        serialized_result = result
        if result is not None and not isinstance(result, str):
            serialized_result = json.dumps(result, ensure_ascii=False)

        normalized_status = normalize_task_status(status)
        for _ in range(self.retry_count):
            try:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE tasks SET status=?, result=?, message=?, cost_time=? WHERE id=?",
                        (normalized_status, serialized_result, message, cost_time, task_id),
                    )
                break
            except sqlite3.OperationalError:
                time.sleep(self.retry_delay)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT status, result, message, cost_time FROM tasks WHERE id=?", (task_id,))
            row = c.fetchone()

        if not row:
            return None

        record = TaskRecord(
            status=row[0],
            result=json.loads(row[1]) if row[1] else None,
            message=row[2],
            cost_time=row[3],
        )
        return record.to_dict()


default_store = SQLiteTaskStore()


def init_db() -> None:
    default_store.init_db()


def create_task(task_id: str) -> None:
    default_store.create_task(task_id)


def update_task(
    task_id: str,
    status: TaskStatus | str,
    result: Optional[Any] = None,
    message: Optional[str] = None,
    cost_time: float = 0,
) -> None:
    default_store.update_task(
        task_id,
        status,
        result=result,
        message=message,
        cost_time=cost_time,
    )


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return default_store.get_task(task_id)


__all__ = [
    "DB_FILE",
    "SQLiteTaskStore",
    "create_task",
    "default_store",
    "get_task",
    "init_db",
    "update_task",
]
