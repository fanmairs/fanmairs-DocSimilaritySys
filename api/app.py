from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from frontend_static import serve_frontend_path

from .routes import config, preview, tasks
from .runtime import ApiRuntime


def create_app() -> FastAPI:
    app = FastAPI(title="智能文档查重系统 (异步队列并发版)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.task_runtime = None

    @app.on_event("startup")
    async def start_task_worker():
        runtime = ApiRuntime()
        app.state.task_runtime = runtime
        runtime.start_worker()

    app.include_router(tasks.router)
    app.include_router(preview.router)
    app.include_router(config.router)

    @app.get("/", include_in_schema=False)
    async def serve_frontend_index():
        return serve_frontend_path()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend_app(full_path: str):
        return serve_frontend_path(full_path)

    return app


app = create_app()


def run() -> None:
    import uvicorn

    host = os.getenv("DOCSIM_HOST", "0.0.0.0")
    port = int(os.getenv("DOCSIM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
