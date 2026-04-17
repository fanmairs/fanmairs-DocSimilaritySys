from __future__ import annotations

import queue
import shutil
import threading
import time
from typing import Callable, Dict, Optional

from ..services.task_runner import TaskRunner


class GpuTaskWorker:
    def __init__(
        self,
        *,
        task_queue: queue.Queue,
        task_runner: TaskRunner,
        update_task_fn: Callable[..., None],
    ):
        self.queue = task_queue
        self.task_runner = task_runner
        self._update_task = update_task_fn
        self._thread: Optional[threading.Thread] = None

    def start(self) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread

        self._thread = threading.Thread(
            target=self.run,
            daemon=True,
            name="docsim-gpu-worker",
        )
        self._thread.start()
        return self._thread

    def run(self) -> None:
        self.task_runner.load()

        while True:
            task: Dict[str, object] = self.queue.get()
            task_id = str(task["id"])
            session_dir = str(task["session_dir"])
            print(f">>> [Task Worker] Processing queued task: {task_id}")
            self._update_task(task_id, "processing")
            start_time = time.time()

            try:
                result_payload = self.task_runner.process(task)
                cost_time = time.time() - start_time
                self._update_task(
                    task_id,
                    "completed",
                    result=result_payload,
                    cost_time=cost_time,
                )
                print(f">>> [Task Worker] Task completed: {task_id} ({cost_time:.2f}s)")
            except Exception as exc:
                import traceback

                traceback.print_exc()
                cost_time = time.time() - start_time
                self._update_task(
                    task_id,
                    "failed",
                    message=str(exc),
                    cost_time=cost_time,
                )
                print(f">>> [Task Worker] Task failed: {task_id} ({cost_time:.2f}s)")
            finally:
                shutil.rmtree(session_dir, ignore_errors=True)
                self.queue.task_done()
