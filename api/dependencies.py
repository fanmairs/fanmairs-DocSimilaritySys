from __future__ import annotations

from fastapi import HTTPException, Request

from .runtime import ApiRuntime


def get_runtime(request: Request) -> ApiRuntime:
    runtime = getattr(request.app.state, "task_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Task runtime is not ready")
    return runtime
