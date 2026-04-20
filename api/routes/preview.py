from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api_bge_helpers import window_recommendation, window_scale_level
from config.settings import get_settings

from ..dependencies import get_runtime
from ..runtime import ApiRuntime


router = APIRouter(prefix="/api", tags=["preview"])


def _create_preview_reader():
    from engines.traditional.system import PlagiarismDetectorSystem

    settings = get_settings()
    return PlagiarismDetectorSystem(
        stopwords_path=settings.stopwords_path,
        lsa_components=settings.lsa_components,
        synonyms_path=settings.synonyms_path,
        semantic_embeddings_path=settings.semantic_embeddings_path,
        semantic_threshold=settings.semantic_threshold,
        semantic_weight=settings.semantic_weight,
    )


@router.post("/bge_window_estimate")
async def bge_window_estimate(
    target_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    body_mode: bool = Form(False),
    runtime: ApiRuntime = Depends(get_runtime),
):
    if not runtime.ensure_ready_for_estimate():
        raise HTTPException(status_code=503, detail="BGE 模型仍在加载，请稍后再估算窗口规模")

    return runtime.estimate_bge_windows(
        target_file=target_file,
        reference_files=reference_files,
        body_mode=body_mode,
        recommendation_fn=window_recommendation,
        scale_level_fn=window_scale_level,
    )


@router.post("/preview_document")
async def preview_document(
    file: UploadFile = File(...),
    runtime: ApiRuntime = Depends(get_runtime),
):
    return runtime.preview_document(
        file=file,
        reader_factory=_create_preview_reader,
    )
