from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

from .common import ScoreLevel, clamp01, resolve_score_level, weighted_average


@dataclass(frozen=True)
class GlobalScoreBreakdown:
    global_score: float
    score_level: ScoreLevel
    raw_score: float
    low_evidence_cap: float
    coverage_signal: float
    evidence_signal: float


def calculate_source_strength(source_entry: Dict[str, object]) -> float:
    risk = clamp01(float(source_entry.get("risk", 0.0)))
    effective_coverage = clamp01(float(source_entry.get("effective_coverage", 0.0)))
    confidence = clamp01(float(source_entry.get("confidence", 0.0)))
    final_score = clamp01(float(source_entry.get("final_score", 0.0)))
    return clamp01(
        0.42 * risk
        + 0.28 * effective_coverage
        + 0.18 * confidence
        + 0.12 * final_score
    )


def calculate_source_support(source_entries: Sequence[Dict[str, object]]) -> float:
    if not source_entries:
        return 0.0

    strengths = [float(item.get("strength", 0.0)) for item in source_entries]
    weights = [
        max(
            float(item.get("effective_coverage", 0.0)),
            float(item.get("weighted_coverage", 0.0)),
            min(0.20, float(item.get("hit_count", 0)) / 20.0),
        )
        for item in source_entries
    ]
    support = weighted_average(strengths, weights)
    if support > 0:
        return support

    top_strengths = sorted(strengths, reverse=True)[: min(3, len(strengths))]
    if not top_strengths:
        return 0.0
    return clamp01(sum(top_strengths) / len(top_strengths))


def calculate_source_diversity(source_entries: Sequence[Dict[str, object]]) -> float:
    if not source_entries:
        return 0.0

    source_count = len(source_entries)
    saturation = 1.0 - math.exp(-float(source_count) / 2.3)
    top_strengths = sorted(
        (float(item.get("strength", 0.0)) for item in source_entries),
        reverse=True,
    )[: min(3, source_count)]
    support = sum(top_strengths) / max(1, len(top_strengths))
    return clamp01(saturation * (0.60 + 0.40 * support))


def calculate_global_score(
    *,
    raw_coverage: float,
    weighted_coverage: float,
    effective_coverage: float,
    confidence: float,
    continuity_top3: float,
    source_support: float,
    source_diversity: float,
) -> GlobalScoreBreakdown:
    raw_coverage = clamp01(raw_coverage)
    weighted_coverage = clamp01(weighted_coverage)
    effective_coverage = clamp01(effective_coverage)
    confidence = clamp01(confidence)
    continuity_top3 = clamp01(continuity_top3)
    source_support = clamp01(source_support)
    source_diversity = clamp01(source_diversity)

    coverage_signal = clamp01(
        0.55 * effective_coverage
        + 0.25 * weighted_coverage
        + 0.20 * continuity_top3
    )
    evidence_signal = clamp01(0.65 * confidence + 0.35 * source_support)
    raw_score = clamp01(
        0.48 * coverage_signal
        + 0.22 * evidence_signal
        + 0.18 * source_diversity
        + 0.12 * raw_coverage
    )

    if effective_coverage < 0.03 and continuity_top3 < 0.02:
        low_evidence_cap = clamp01(0.24 + 0.20 * evidence_signal + 0.18 * source_diversity)
    elif effective_coverage < 0.06 and confidence < 0.55:
        low_evidence_cap = clamp01(0.36 + 0.16 * source_support)
    else:
        low_evidence_cap = 1.0

    global_score = clamp01(min(raw_score, low_evidence_cap))
    return GlobalScoreBreakdown(
        global_score=global_score,
        score_level=resolve_score_level(global_score),
        raw_score=raw_score,
        low_evidence_cap=low_evidence_cap,
        coverage_signal=coverage_signal,
        evidence_signal=evidence_signal,
    )
