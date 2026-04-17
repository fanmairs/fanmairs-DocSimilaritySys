from __future__ import annotations

import os
import shutil
import uuid
from typing import List


def _safe_upload_name(upload_file, fallback: str) -> str:
    filename = getattr(upload_file, "filename", None) or fallback
    filename = str(filename).replace("\\", "/").split("/")[-1].strip()
    return filename or fallback


def copy_upload(upload_file, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file_obj:
        shutil.copyfileobj(upload_file.file, file_obj)


def save_task_uploads(
    *,
    temp_dir: str,
    task_id: str,
    target_file,
    reference_files: List[object],
) -> tuple[str, List[str], str]:
    session_dir = os.path.join(temp_dir, task_id)
    os.makedirs(session_dir, exist_ok=True)

    target_name = _safe_upload_name(target_file, "target")
    target_path = os.path.join(session_dir, f"target_{target_name}")
    copy_upload(target_file, target_path)

    ref_paths: List[str] = []
    for index, ref_file in enumerate(reference_files, start=1):
        ref_name = _safe_upload_name(ref_file, f"reference_{index}")
        ref_path = os.path.join(session_dir, f"ref_{ref_name}")
        copy_upload(ref_file, ref_path)
        ref_paths.append(ref_path)

    return target_path, ref_paths, session_dir


def save_estimate_uploads(
    *,
    estimate_dir: str,
    target_file,
    reference_files: List[object],
) -> tuple[str, List[str]]:
    os.makedirs(estimate_dir, exist_ok=True)

    target_name = _safe_upload_name(target_file, "target")
    target_path = os.path.join(estimate_dir, f"target_{target_name}")
    copy_upload(target_file, target_path)

    ref_paths: List[str] = []
    for index, ref_file in enumerate(reference_files, start=1):
        ref_name = _safe_upload_name(ref_file, f"reference_{index}")
        ref_path = os.path.join(estimate_dir, f"ref_{ref_name}")
        copy_upload(ref_file, ref_path)
        ref_paths.append(ref_path)

    return target_path, ref_paths


def save_preview_upload(temp_dir: str, upload_file) -> str:
    filename = _safe_upload_name(upload_file, "preview")
    temp_path = os.path.join(temp_dir, f"preview_{uuid.uuid4().hex[:8]}_{filename}")
    copy_upload(upload_file, temp_path)
    return temp_path
