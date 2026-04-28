from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api_bge_helpers import (
    BGE_STRATEGY_COARSE,
    parse_coarse_config_payload,
    resolve_bge_strategy,
)

from ..dependencies import get_runtime
from ..runtime import ApiRuntime


VALID_BERT_PROFILES = {"strict", "balanced", "recall"}
MIN_LSA_COMPONENTS = 1
MAX_LSA_COMPONENTS = 12

router = APIRouter(prefix="/api", tags=["tasks"])


def _normalize_bert_profile(value: str | None) -> str:
    normalized = (value or "balanced").strip().lower()
    return normalized if normalized in VALID_BERT_PROFILES else "balanced"


def _normalize_lsa_components(value: int | str | None) -> int:
    try:
        components = int(value)
    except (TypeError, ValueError):
        components = 3
    return max(MIN_LSA_COMPONENTS, min(components, MAX_LSA_COMPONENTS))


@router.post("/submit_task")
async def submit_task(
    target_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    mode: str = Form("bert"),
    body_mode: bool = Form(False),
    bert_profile: str = Form("balanced"),
    bge_strategy: str = Form(BGE_STRATEGY_COARSE),
    lsa_components: int = Form(3),
    coarse_config: str = Form(""),
    runtime: ApiRuntime = Depends(get_runtime),
):
    safe_bge_strategy = resolve_bge_strategy(bge_strategy)
    safe_coarse_config = (
        parse_coarse_config_payload(coarse_config)
        if safe_bge_strategy == BGE_STRATEGY_COARSE
        else None
    )

    return runtime.submit_task(
        target_file=target_file,
        reference_files=reference_files,
        mode=mode,
        body_mode=body_mode,
        bert_profile=_normalize_bert_profile(bert_profile),
        bge_strategy=safe_bge_strategy,
        lsa_components=_normalize_lsa_components(lsa_components),
        coarse_config=safe_coarse_config,
    )


@router.get("/task_status/{task_id}")
async def check_task_status(
    task_id: str,
    runtime: ApiRuntime = Depends(get_runtime),
):
    task_info = runtime.get_status(task_id)
    if not task_info:
        return {"status": "error", "message": "任务不存在或已过期"}

    return {
        "status": "success",
        "task_status": task_info["status"],
        "data": task_info["result"],
        "message": task_info["message"],
        "cost_time": task_info["cost_time"],
    }
