"""Shared scoring utilities for traditional and semantic detection."""

from .common import (
    ScoreLevel,
    build_score_metadata,
    clamp01,
    resolve_score_level,
    weighted_average,
)
from .coarse import calculate_paragraph_hotspot, compose_coarse_score
from .global_summary import calculate_global_score
from .semantic import (
    calculate_semantic_excess,
    calculate_semantic_pair_score,
    calculate_semantic_risk_score,
)
from .traditional import calculate_risk_score, fuse_similarity_scores
from .window import resolve_outlier_metrics, score_window_candidate, select_topk_indices

__all__ = [
    "ScoreLevel",
    "build_score_metadata",
    "calculate_global_score",
    "calculate_paragraph_hotspot",
    "calculate_risk_score",
    "calculate_semantic_excess",
    "calculate_semantic_pair_score",
    "calculate_semantic_risk_score",
    "clamp01",
    "compose_coarse_score",
    "fuse_similarity_scores",
    "resolve_outlier_metrics",
    "resolve_score_level",
    "score_window_candidate",
    "select_topk_indices",
    "weighted_average",
]
