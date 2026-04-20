from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from .models import TaskStatus


class TaskRepository(Protocol):
    def init_db(self) -> None:
        ...

    def create_task(self, task_id: str) -> None:
        ...

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | str,
        result: Optional[Any] = None,
        message: Optional[str] = None,
        cost_time: float = 0,
    ) -> None:
        ...

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        ...
