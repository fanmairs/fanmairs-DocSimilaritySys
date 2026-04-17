from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import queue
import shutil
import uuid
from typing import Callable, Dict, List, Optional

from task_store import create_task, get_task, init_db, update_task

from .services.task_runner import TaskRunner
from .services.uploads import (
    copy_upload,
    save_estimate_uploads,
    save_preview_upload,
    save_task_uploads,
)
from .workers.gpu_worker import GpuTaskWorker


@dataclass(frozen=True)
class QueuedTask:
    id: str
    target_path: str
    ref_paths: List[str]
    mode: str
    body_mode: bool
    bert_profile: str
    bge_strategy: str
    coarse_config: Optional[Dict[str, object]]
    session_dir: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ApiRuntime:
    def __init__(
        self,
        *,
        temp_dir: str = "temp_uploads",
        task_runner: Optional[TaskRunner] = None,
        task_queue: Optional[queue.Queue] = None,
        task_store_init: Callable[[], None] = init_db,
        create_task_fn: Callable[[str], None] = create_task,
        get_task_fn: Callable[[str], Optional[Dict[str, object]]] = get_task,
        update_task_fn: Callable[..., None] = update_task,
    ):
        self.temp_dir = temp_dir
        self.task_runner = task_runner or TaskRunner()
        self.queue = task_queue or queue.Queue()
        self._create_task = create_task_fn
        self._get_task = get_task_fn
        self._update_task = update_task_fn
        self.worker = GpuTaskWorker(
            task_queue=self.queue,
            task_runner=self.task_runner,
            update_task_fn=self._update_task,
        )

        os.makedirs(self.temp_dir, exist_ok=True)
        task_store_init()

    @property
    def processor(self) -> TaskRunner:
        return self.task_runner

    def start_worker(self):
        return self.worker.start()

    def submit_task(
        self,
        *,
        target_file,
        reference_files: List[object],
        mode: str,
        body_mode: bool,
        bert_profile: str,
        bge_strategy: str,
        coarse_config: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        task_id = str(uuid.uuid4())
        session_dir: Optional[str] = None

        try:
            target_path, ref_paths, session_dir = save_task_uploads(
                temp_dir=self.temp_dir,
                task_id=task_id,
                target_file=target_file,
                reference_files=reference_files,
            )
            self._create_task(task_id)

            queued_task = QueuedTask(
                id=task_id,
                target_path=target_path,
                ref_paths=ref_paths,
                mode=mode,
                body_mode=body_mode,
                bert_profile=bert_profile,
                bge_strategy=bge_strategy,
                coarse_config=coarse_config,
                session_dir=session_dir,
            )
            self.queue.put(queued_task.to_dict())
        except Exception:
            if session_dir:
                shutil.rmtree(session_dir, ignore_errors=True)
            raise

        queue_length = self.queue.qsize()
        return {
            "status": "success",
            "task_id": task_id,
            "queue_length": queue_length,
            "message": (
                f"任务已提交，前面还有 {queue_length - 1} 个任务在排队。"
                if queue_length > 1
                else "任务已提交，即将开始计算。"
            ),
        }

    def get_status(self, task_id: str) -> Optional[Dict[str, object]]:
        return self._get_task(task_id)

    def ensure_ready_for_estimate(self) -> bool:
        return self.task_runner.is_ready()

    def estimate_bge_windows(
        self,
        *,
        target_file,
        reference_files: List[object],
        body_mode: bool,
        recommendation_fn,
        scale_level_fn,
    ) -> Dict[str, object]:
        estimate_id = uuid.uuid4().hex[:10]
        estimate_dir = os.path.join(self.temp_dir, f"estimate_{estimate_id}")

        try:
            target_path, ref_paths = save_estimate_uploads(
                estimate_dir=estimate_dir,
                target_file=target_file,
                reference_files=reference_files,
            )

            target_text = self.task_runner.read_document(target_path, body_mode=body_mode)
            target_window_count = self.task_runner.estimate_window_count(target_text)

            reference_summaries = []
            total_reference_windows = 0
            for ref_path in ref_paths:
                ref_text = self.task_runner.read_document(ref_path, body_mode=body_mode)
                window_count = self.task_runner.estimate_window_count(ref_text)
                total_reference_windows += window_count
                reference_summaries.append(
                    {
                        "file": os.path.basename(ref_path).replace("ref_", ""),
                        "window_count": int(window_count),
                    }
                )

            full_pair_count = int(target_window_count * total_reference_windows)
            return {
                "status": "success",
                "target_window_count": int(target_window_count),
                "reference_window_count": int(total_reference_windows),
                "reference_count": int(len(ref_paths)),
                "average_reference_windows": float(
                    total_reference_windows / len(ref_paths)
                    if ref_paths
                    else 0.0
                ),
                "full_pair_count": full_pair_count,
                "scale_level": scale_level_fn(full_pair_count),
                "recommendation": recommendation_fn(full_pair_count, len(ref_paths)),
                "references": reference_summaries[:20],
            }
        finally:
            shutil.rmtree(estimate_dir, ignore_errors=True)

    def preview_document(
        self,
        *,
        file,
        reader_factory,
    ) -> Dict[str, object]:
        temp_path = save_preview_upload(self.temp_dir, file)
        try:
            reader = reader_factory()
            content = reader.read_document(temp_path, preview_mode=True)
            return {"status": "success", "content": content}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def copy_upload(upload_file, path: str) -> None:
        copy_upload(upload_file, path)


TaskRuntime = ApiRuntime
