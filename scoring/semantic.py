from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .common import clamp01


@dataclass(frozen=True)
class SemanticPairScore:
    final_score: float
    effective_coverage: float
    semantic_signal: float
    evidence_score: float
    continuity_bonus: float
    low_evidence_cap: float

    def as_tuple(self):
        return (
            self.final_score,
            self.effective_coverage,
            self.semantic_signal,
            self.evidence_score,
            self.continuity_bonus,
            self.low_evidence_cap,
        )


@dataclass(frozen=True)
class SemanticRiskScore:
    risk_score: float
    base_score: float
    gate: float


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return float(1.0 / (1.0 + exp_term))

    exp_term = math.exp(value)
    return float(exp_term / (1.0 + exp_term))


def calculate_semantic_excess(doc_semantic: float, semantic_floor: float) -> float:
    return clamp01(
        (clamp01(doc_semantic) - float(semantic_floor))
        / max(1e-6, 1.0 - float(semantic_floor))
    )


def calculate_semantic_base_score(
    profile_cfg: Dict,
    semantic_excess: float,
    weighted_coverage: float,
    confidence: float,
) -> float:
    weights = profile_cfg["score_weights"]
    return clamp01(
        float(weights["doc_semantic"]) * clamp01(semantic_excess)
        + float(weights["coverage"]) * clamp01(weighted_coverage)
        + float(weights["confidence"]) * clamp01(confidence)
    )


def calculate_semantic_risk_score(
    profile_cfg: Dict,
    semantic_excess: float,
    weighted_coverage: float,
    confidence: float,
) -> SemanticRiskScore:
    weighted_coverage = clamp01(weighted_coverage)
    confidence = clamp01(confidence)
    base_score = calculate_semantic_base_score(
        profile_cfg,
        semantic_excess,
        weighted_coverage,
        confidence,
    )

    gate_cfg = profile_cfg["score_gate"]
    evidence_strength = (weighted_coverage * confidence) ** 0.5
    gate = clamp01(
        float(gate_cfg["base"])
        + float(gate_cfg["coverage"]) * weighted_coverage
        + float(gate_cfg["confidence"]) * confidence
        + float(gate_cfg.get("evidence", 0.0)) * evidence_strength
    )

    risk_score = clamp01(base_score * gate)

    if weighted_coverage < float(gate_cfg["low_cov_th"]):
        low_cov_cap = gate_cfg.get("low_cov_cap")
        if low_cov_cap is not None:
            risk_score = min(risk_score, float(low_cov_cap))
    elif (
        weighted_coverage < float(gate_cfg["mid_cov_th"])
        and confidence < float(gate_cfg["mid_conf_th"])
    ):
        risk_score = min(risk_score, float(gate_cfg["mid_cov_cap"]))
    elif (
        weighted_coverage < float(gate_cfg.get("topic_cov_th", 0.0))
        and confidence < float(gate_cfg.get("topic_conf_th", 1.0))
    ):
        risk_score = min(risk_score, float(gate_cfg.get("topic_cap", 1.0)))

    if (
        weighted_coverage < float(gate_cfg.get("low_evidence_cov_th", 0.0))
        and confidence < float(gate_cfg.get("low_evidence_conf_th", 1.0))
    ):
        risk_score = min(risk_score, float(gate_cfg.get("low_evidence_cap", 1.0)))

    return SemanticRiskScore(
        risk_score=clamp01(risk_score),
        base_score=base_score,
        gate=gate,
    )


def calculate_semantic_pair_score(
    profile_cfg: Dict,
    *,
    effective_coverage: float,
    confidence: float,
    doc_semantic: float,
    paragraph_semantic: float,
    longest_run_ratio: float,
    top3_run_ratio: float,
) -> SemanticPairScore:
    effective_coverage = clamp01(effective_coverage)
    confidence = clamp01(confidence)
    longest_run_ratio = clamp01(longest_run_ratio)
    top3_run_ratio = clamp01(top3_run_ratio)

    final_cfg = profile_cfg.get("final_score", {})
    semantic_weight = float(final_cfg.get("semantic_weight", 0.30))
    evidence_weight = float(final_cfg.get("evidence_weight", 0.70))
    semantic_center = float(final_cfg.get("semantic_center", 0.80))
    semantic_scale = max(1e-6, float(final_cfg.get("semantic_scale", 0.08)))
    coverage_gain = max(1e-6, float(final_cfg.get("coverage_gain", 8.0)))
    low_evidence_cap_base = float(final_cfg.get("low_evidence_cap_base", 0.14))
    low_evidence_cap_gain = float(final_cfg.get("low_evidence_cap_gain", 0.20))
    continuity_boost = float(final_cfg.get("continuity_boost", 0.06))

    semantic_input = clamp01(0.60 * doc_semantic + 0.40 * paragraph_semantic)
    semantic_signal = sigmoid((semantic_input - semantic_center) / semantic_scale)

    coverage_core = clamp01(
        0.55 * effective_coverage
        + 0.25 * longest_run_ratio
        + 0.20 * top3_run_ratio
    )
    evidence_score = clamp01(1.0 - math.exp(-coverage_gain * coverage_core))
    confidence_scale = 0.72 + 0.28 * (confidence ** 0.85)
    evidence_score = clamp01(evidence_score * confidence_scale)

    continuity_signal = clamp01(max(longest_run_ratio * 10.0, top3_run_ratio * 5.0))
    continuity_bonus = continuity_boost * continuity_signal * confidence

    final_score = (
        semantic_weight * semantic_signal
        + evidence_weight * evidence_score
        + continuity_bonus
    )

    if effective_coverage < 0.03 and longest_run_ratio < 0.01:
        low_evidence_cap = low_evidence_cap_base + low_evidence_cap_gain * semantic_signal
        final_score = min(final_score, low_evidence_cap)
    else:
        low_evidence_cap = 1.0

    return SemanticPairScore(
        final_score=clamp01(final_score),
        effective_coverage=effective_coverage,
        semantic_signal=float(semantic_signal),
        evidence_score=float(evidence_score),
        continuity_bonus=float(continuity_bonus),
        low_evidence_cap=float(low_evidence_cap),
    )
