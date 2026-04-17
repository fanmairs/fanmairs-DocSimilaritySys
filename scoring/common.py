from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence


ScoreLevel = Literal["low", "medium", "high"]


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0

    total_weight = float(sum(max(0.0, float(weight)) for weight in weights))
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(
        float(value) * max(0.0, float(weight))
        for value, weight in zip(values, weights)
    )
    return clamp01(weighted_sum / total_weight)


def resolve_score_level(
    score: float,
    *,
    medium_threshold: float = 0.35,
    high_threshold: float = 0.70,
) -> ScoreLevel:
    normalized = clamp01(score)
    if normalized >= high_threshold:
        return "high"
    if normalized >= medium_threshold:
        return "medium"
    return "low"


def build_score_metadata(
    *,
    engine: str,
    score: float,
    risk_score: Optional[float] = None,
    coverage: Optional[float] = None,
    confidence: Optional[float] = None,
) -> Dict[str, object]:
    normalized_score = clamp01(score)
    normalized_risk = clamp01(risk_score if risk_score is not None else normalized_score)
    metadata: Dict[str, object] = {
        "score_engine": engine,
        "score": normalized_score,
        "score_level": resolve_score_level(normalized_score),
        "risk_score_normalized": normalized_risk,
        "risk_level": resolve_score_level(normalized_risk),
    }
    if coverage is not None:
        metadata["score_coverage"] = clamp01(coverage)
    if confidence is not None:
        metadata["score_confidence"] = clamp01(confidence)
    return metadata
