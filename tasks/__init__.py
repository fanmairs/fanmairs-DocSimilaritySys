"""Task models, persistence, and compatibility exports."""

from .models import TaskRecord, TaskStatus
from .repository import TaskRepository
from .sqlite_store import (
    DB_FILE,
    SQLiteTaskStore,
    create_task,
    default_store,
    get_task,
    init_db,
    update_task,
)


def __getattr__(name: str):
    if name in {"TaskProcessor", "TaskRunner"}:
        from .processor import TaskProcessor, TaskRunner

        return {"TaskProcessor": TaskProcessor, "TaskRunner": TaskRunner}[name]
    if name in {"ApiRuntime", "QueuedTask", "TaskRuntime"}:
        from .runtime import ApiRuntime, QueuedTask, TaskRuntime

        return {
            "ApiRuntime": ApiRuntime,
            "QueuedTask": QueuedTask,
            "TaskRuntime": TaskRuntime,
        }[name]
    raise AttributeError(f"module 'tasks' has no attribute {name!r}")


__all__ = [
    "DB_FILE",
    "SQLiteTaskStore",
    "TaskRecord",
    "TaskRepository",
    "TaskStatus",
    "create_task",
    "default_store",
    "get_task",
    "init_db",
    "update_task",
    "ApiRuntime",
    "QueuedTask",
    "TaskProcessor",
    "TaskRunner",
    "TaskRuntime",
]
