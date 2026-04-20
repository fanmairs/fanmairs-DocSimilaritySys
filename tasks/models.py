from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


INTERRUPTED_ON_RESTART_MESSAGE = "服务器重启，排队或计算中的任务已中断"


@dataclass(frozen=True)
class TaskRecord:
    status: str
    result: Optional[Any]
    message: Optional[str]
    cost_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "result": self.result,
            "message": self.message,
            "cost_time": self.cost_time,
        }


def normalize_task_status(status: TaskStatus | str) -> str:
    if isinstance(status, TaskStatus):
        return status.value
    return str(status)
