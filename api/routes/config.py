from __future__ import annotations

from fastapi import APIRouter

from engines.semantic.coarse_retrieval import CoarseRetrievalConfig


router = APIRouter(prefix="/api", tags=["config"])


@router.get("/coarse_config_defaults")
async def coarse_config_defaults():
    return {
        "status": "success",
        "defaults": CoarseRetrievalConfig().normalized().to_dict(),
    }
