from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .bge_backend import DeepSemanticEngine


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0

    total_weight = float(sum(max(0.0, weight) for weight in weights))
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(
        float(value) * max(0.0, float(weight))
        for value, weight in zip(values, weights)
    )
    return _clamp01(weighted_sum / total_weight)


def _collect_interval_parts(parts: Sequence[Dict]) -> List[Dict]:
    interval_parts: List[Dict] = []
    for part in parts:
        start = part.get("target_start")
        end = part.get("target_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            interval_parts.append(dict(part))
    return interval_parts


class GlobalEvidenceAggregator:
    def __init__(self, semantic_engine: DeepSemanticEngine):
        self.semantic_engine = semantic_engine

    def _target_length(self, target_text: str) -> int:
        normalized = DeepSemanticEngine._normalize_text(target_text or "")
        return max(1, len(normalized))

    def _source_strength(self, source_entry: Dict[str, float]) -> float:
        risk = _clamp01(float(source_entry.get("risk", 0.0)))
        effective_coverage = _clamp01(float(source_entry.get("effective_coverage", 0.0)))
        confidence = _clamp01(float(source_entry.get("confidence", 0.0)))
        final_score = _clamp01(float(source_entry.get("final_score", 0.0)))
        return _clamp01(
            0.42 * risk
            + 0.28 * effective_coverage
            + 0.18 * confidence
            + 0.12 * final_score
        )

    def _build_source_entry(self, item: Dict[str, object], target_len: int) -> Optional[Dict[str, object]]:
        parts = list(item.get("plagiarized_parts") or [])
        hit_count = int(item.get("sim_bert_hits", len(parts)) or 0)
        if not parts and hit_count <= 0:
            return None

        interval_parts = _collect_interval_parts(parts)
        raw_coverage = 0.0
        weighted_coverage = 0.0
        if interval_parts:
            raw_coverage = self.semantic_engine._calculate_raw_coverage(interval_parts, target_len)
            weighted_coverage = self.semantic_engine._calculate_coverage(interval_parts, target_len)

        confidence_input = interval_parts if interval_parts else parts
        confidence = self.semantic_engine._calculate_match_confidence(confidence_input) if confidence_input else 0.0
        effective_coverage = self.semantic_engine._calculate_effective_coverage(
            raw_coverage,
            weighted_coverage,
            confidence,
        )

        source_entry: Dict[str, object] = {
            "file": item.get("file", "unknown"),
            "risk": _clamp01(float(item.get("sim_bert_risk", item.get("sim_bert", 0.0)))),
            "final_score": _clamp01(float(item.get("sim_bert", item.get("sim_bert_risk", 0.0)))),
            "raw_coverage": float(raw_coverage),
            "weighted_coverage": float(weighted_coverage),
            "effective_coverage": float(effective_coverage),
            "confidence": float(confidence),
            "hit_count": hit_count,
            "interval_hit_count": len(interval_parts),
            "interval_parts": interval_parts,
        }
        source_entry["strength"] = self._source_strength(source_entry)
        return source_entry

    def _calculate_source_support(self, source_entries: Sequence[Dict[str, object]]) -> float:
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
        support = _weighted_average(strengths, weights)
        if support > 0:
            return support

        top_strengths = sorted(strengths, reverse=True)[: min(3, len(strengths))]
        if not top_strengths:
            return 0.0
        return _clamp01(sum(top_strengths) / len(top_strengths))

    def _calculate_source_diversity(self, source_entries: Sequence[Dict[str, object]]) -> float:
        if not source_entries:
            return 0.0

        source_count = len(source_entries)
        saturation = 1.0 - math.exp(-float(source_count) / 2.3)
        top_strengths = sorted(
            (float(item.get("strength", 0.0)) for item in source_entries),
            reverse=True,
        )[: min(3, source_count)]
        support = sum(top_strengths) / max(1, len(top_strengths))
        return _clamp01(saturation * (0.60 + 0.40 * support))

    def aggregate(
        self,
        target_text: str,
        verified_results: Sequence[Dict[str, object]],
        *,
        bert_profile: str = "balanced",
        reference_count: Optional[int] = None,
        candidate_count: Optional[int] = None,
        retrieval_strategy: str = "coarse_then_fine",
    ) -> Dict[str, object]:
        target_len = self._target_length(target_text)
        source_entries = []
        all_interval_parts: List[Dict] = []
        total_hits = 0
        interval_hits = 0

        for item in verified_results:
            source_entry = self._build_source_entry(item, target_len)
            if not source_entry:
                continue

            source_entries.append(source_entry)
            interval_parts = list(source_entry["interval_parts"])
            all_interval_parts.extend(interval_parts)
            total_hits += int(source_entry["hit_count"])
            interval_hits += int(source_entry["interval_hit_count"])

        source_entries = sorted(
            source_entries,
            key=lambda item: float(item.get("strength", 0.0)),
            reverse=True,
        )

        raw_coverage = 0.0
        weighted_coverage = 0.0
        confidence = 0.0
        effective_coverage = 0.0
        continuity = {
            "longest_run_ratio": 0.0,
            "top3_run_ratio": 0.0,
            "merged_hit_count": 0,
        }
        if all_interval_parts:
            raw_coverage = self.semantic_engine._calculate_raw_coverage(all_interval_parts, target_len)
            weighted_coverage = self.semantic_engine._calculate_coverage(all_interval_parts, target_len)
            confidence = self.semantic_engine._calculate_match_confidence(all_interval_parts)
            effective_coverage = self.semantic_engine._calculate_effective_coverage(
                raw_coverage,
                weighted_coverage,
                confidence,
            )
            continuity = self.semantic_engine._calculate_continuity_features(all_interval_parts, target_len)
        elif source_entries:
            confidence = _weighted_average(
                [float(item.get("confidence", 0.0)) for item in source_entries],
                [
                    max(float(item.get("strength", 0.0)), 0.01)
                    for item in source_entries
                ],
            ) * 0.50

        source_support = self._calculate_source_support(source_entries)
        source_diversity = self._calculate_source_diversity(source_entries)

        coverage_signal = _clamp01(
            0.55 * effective_coverage
            + 0.25 * weighted_coverage
            + 0.20 * float(continuity["top3_run_ratio"])
        )
        evidence_signal = _clamp01(0.65 * confidence + 0.35 * source_support)
        raw_score = _clamp01(
            0.48 * coverage_signal
            + 0.22 * evidence_signal
            + 0.18 * source_diversity
            + 0.12 * raw_coverage
        )

        if effective_coverage < 0.03 and float(continuity["top3_run_ratio"]) < 0.02:
            low_evidence_cap = _clamp01(0.24 + 0.20 * evidence_signal + 0.18 * source_diversity)
        elif effective_coverage < 0.06 and confidence < 0.55:
            low_evidence_cap = _clamp01(0.36 + 0.16 * source_support)
        else:
            low_evidence_cap = 1.0

        global_score = _clamp01(min(raw_score, low_evidence_cap))
        score_level = "low"
        if global_score >= 0.70:
            score_level = "high"
        elif global_score >= 0.35:
            score_level = "medium"

        top_sources = [
            {
                "file": item["file"],
                "source_strength": float(item["strength"]),
                "risk": float(item["risk"]),
                "effective_coverage": float(item["effective_coverage"]),
                "confidence": float(item["confidence"]),
                "hit_count": int(item["hit_count"]),
                "interval_hit_count": int(item["interval_hit_count"]),
            }
            for item in source_entries[:3]
        ]

        return {
            "retrieval_stage": "global_summary",
            "retrieval_strategy": retrieval_strategy,
            "bert_profile": bert_profile,
            "global_score": float(global_score),
            "global_score_level": score_level,
            "global_score_raw": float(raw_score),
            "global_low_evidence_cap": float(low_evidence_cap),
            "global_coverage_raw": float(raw_coverage),
            "global_coverage_weighted": float(weighted_coverage),
            "global_coverage_effective": float(effective_coverage),
            "global_confidence": float(confidence),
            "global_continuity_longest": float(continuity["longest_run_ratio"]),
            "global_continuity_top3": float(continuity["top3_run_ratio"]),
            "global_dedup_span_count": int(continuity["merged_hit_count"]),
            "global_source_diversity": float(source_diversity),
            "global_source_support": float(source_support),
            "global_verified_source_count": int(len(source_entries)),
            "global_supported_source_count": int(sum(1 for item in source_entries if float(item.get("effective_coverage", 0.0)) > 0.0)),
            "global_candidate_count": int(candidate_count if candidate_count is not None else len(verified_results)),
            "global_reference_count": int(reference_count if reference_count is not None else len(verified_results)),
            "global_hit_count": int(total_hits),
            "global_interval_hit_count": int(interval_hits),
            "global_target_length": int(target_len),
            "top_sources": top_sources,
        }
