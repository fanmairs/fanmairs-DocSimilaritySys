import json
import sqlite3
import time
from typing import Any, Dict, Optional


DB_FILE = "tasks.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_FILE)
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
        "UPDATE tasks SET status='failed', message='服务器重启，排队或计算中的任务已中断' "
        "WHERE status IN ('pending', 'processing')"
    )
    c.execute("DELETE FROM tasks WHERE created_at <= datetime('now', '-3 days')")

    conn.commit()
    conn.close()


def create_task(task_id: str) -> None:
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, status) VALUES (?, ?)", (task_id, "pending"))
    conn.commit()
    conn.close()


def update_task(
    task_id: str,
    status: str,
    result: Optional[Any] = None,
    message: Optional[str] = None,
    cost_time: float = 0,
) -> None:
    serialized_result = result
    if result is not None and not isinstance(result, str):
        serialized_result = json.dumps(result, ensure_ascii=False)

    for _ in range(5):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=5.0)
            c = conn.cursor()
            c.execute(
                "UPDATE tasks SET status=?, result=?, message=?, cost_time=? WHERE id=?",
                (status, serialized_result, message, cost_time, task_id),
            )
            conn.commit()
            conn.close()
            break
        except sqlite3.OperationalError:
            time.sleep(0.2)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    c = conn.cursor()
    c.execute("SELECT status, result, message, cost_time FROM tasks WHERE id=?", (task_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "status": row[0],
            "result": json.loads(row[1]) if row[1] else None,
            "message": row[2],
            "cost_time": row[3],
        }
    return None
