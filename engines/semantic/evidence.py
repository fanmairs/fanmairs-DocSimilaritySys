"""Compatibility layer for semantic evidence helpers.

New code should import from ``evidence.metrics``.
"""

from typing import Dict, List, Tuple

from evidence.metrics import (
    calculate_continuity_features as _calculate_continuity_features,
    calculate_coverage as _calculate_coverage,
    calculate_effective_coverage as _calculate_effective_coverage,
    calculate_match_confidence as _calculate_match_confidence,
    calculate_raw_coverage as _calculate_raw_coverage,
    collect_target_intervals as _collect_target_intervals,
)
from scoring.semantic import calculate_semantic_pair_score


def collect_target_intervals(engine, plagiarized_parts: List[Dict], target_len: int) -> List[Tuple[int, int]]:
    return _collect_target_intervals(plagiarized_parts, target_len)


def calculate_raw_coverage(engine, plagiarized_parts: List[Dict], target_len: int) -> float:
    return _calculate_raw_coverage(plagiarized_parts, target_len)


def calculate_coverage(engine, plagiarized_parts: List[Dict], target_len: int) -> float:
    return _calculate_coverage(plagiarized_parts, target_len)


def calculate_match_confidence(engine, plagiarized_parts: List[Dict]) -> float:
    return _calculate_match_confidence(plagiarized_parts)


def calculate_effective_coverage(
    engine,
    raw_coverage: float,
    weighted_coverage: float,
    confidence: float,
) -> float:
    return _calculate_effective_coverage(raw_coverage, weighted_coverage, confidence)


def calculate_continuity_features(
    engine,
    plagiarized_parts: List[Dict],
    target_len: int,
) -> Dict[str, float]:
    return _calculate_continuity_features(plagiarized_parts, target_len)


def calculate_realistic_score(
    engine,
    profile_name: str,
    profile_cfg: Dict,
    raw_coverage: float,
    weighted_coverage: float,
    confidence: float,
    doc_semantic: float,
    paragraph_semantic: float,
    hit_count: int,
    longest_run_ratio: float,
    top3_run_ratio: float,
) -> Tuple[float, float, float, float, float, float]:
    effective_coverage = engine._calculate_effective_coverage(
        raw_coverage,
        weighted_coverage,
        confidence,
    )
    return calculate_semantic_pair_score(
        profile_cfg,
        effective_coverage=effective_coverage,
        confidence=confidence,
        doc_semantic=doc_semantic,
        paragraph_semantic=paragraph_semantic,
        longest_run_ratio=longest_run_ratio,
        top3_run_ratio=top3_run_ratio,
    ).as_tuple()
