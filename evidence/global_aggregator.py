from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from scoring.common import clamp01, weighted_average
from scoring.global_summary import (
    calculate_global_score,
    calculate_source_diversity,
    calculate_source_strength,
    calculate_source_support,
)

from .adapters import normalize_evidence_spans
from .metrics import (
    collect_interval_parts,
    calculate_continuity_features,
    calculate_coverage,
    calculate_effective_coverage,
    calculate_match_confidence,
    calculate_raw_coverage,
)


def _detect_engine_name(item: Dict[str, object]) -> str:
    if "sim_bert" in item or "sim_bert_risk" in item:
        return "semantic"
    if "sim_hybrid" in item or "risk_score" in item:
        return "traditional"
    return str(item.get("engine", "unknown") or "unknown")


class GlobalEvidenceAggregator:
    def __init__(
        self,
        semantic_engine=None,
        *,
        target_normalizer: Optional[Callable[[str], str]] = None,
    ):
        if target_normalizer is None and semantic_engine is not None:
            target_normalizer = getattr(semantic_engine, "_normalize_text", None)
        self.target_normalizer = target_normalizer or (lambda text: text or "")

    def _target_length(self, target_text: str) -> int:
        normalized = self.target_normalizer(target_text or "")
        return max(1, len(normalized))

    def _source_strength(self, source_entry: Dict[str, float]) -> float:
        return calculate_source_strength(source_entry)

    def _build_source_entry(self, item: Dict[str, object], target_len: int) -> Optional[Dict[str, object]]:
        source_file = item.get("file", "unknown")
        engine_name = _detect_engine_name(item)
        parts = normalize_evidence_spans(
            item.get("plagiarized_parts") or [],
            engine=engine_name,
            source=str(source_file),
        )
        hit_count = int(item.get("sim_bert_hits", item.get("hit_count", len(parts))) or 0)
        if not parts and hit_count <= 0:
            return None

        interval_parts = collect_interval_parts(parts)
        raw_coverage = 0.0
        weighted_coverage = 0.0
        if interval_parts:
            raw_coverage = calculate_raw_coverage(interval_parts, target_len)
            weighted_coverage = calculate_coverage(interval_parts, target_len)

        confidence_input = interval_parts if interval_parts else parts
        confidence = calculate_match_confidence(confidence_input) if confidence_input else 0.0
        effective_coverage = calculate_effective_coverage(
            raw_coverage,
            weighted_coverage,
            confidence,
        )

        source_entry: Dict[str, object] = {
            "file": source_file,
            "engine": engine_name,
            "risk": clamp01(
                float(item.get("sim_bert_risk", item.get("risk_score", item.get("sim_bert", item.get("sim_hybrid", 0.0)))))
            ),
            "final_score": clamp01(
                float(item.get("sim_bert", item.get("sim_hybrid", item.get("sim_bert_risk", item.get("risk_score", 0.0)))))
            ),
            "raw_coverage": float(raw_coverage),
            "weighted_coverage": float(weighted_coverage),
            "effective_coverage": float(effective_coverage),
            "confidence": float(confidence),
            "hit_count": hit_count,
            "interval_hit_count": len(interval_parts),
            "interval_parts": interval_parts,
            "evidence_parts": parts,
        }
        source_entry["strength"] = self._source_strength(source_entry)
        return source_entry

    def _calculate_source_support(self, source_entries: Sequence[Dict[str, object]]) -> float:
        return calculate_source_support(source_entries)

    def _calculate_source_diversity(self, source_entries: Sequence[Dict[str, object]]) -> float:
        return calculate_source_diversity(source_entries)

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
            raw_coverage = calculate_raw_coverage(all_interval_parts, target_len)
            weighted_coverage = calculate_coverage(all_interval_parts, target_len)
            confidence = calculate_match_confidence(all_interval_parts)
            effective_coverage = calculate_effective_coverage(
                raw_coverage,
                weighted_coverage,
                confidence,
            )
            continuity = calculate_continuity_features(all_interval_parts, target_len)
        elif source_entries:
            confidence = weighted_average(
                [float(item.get("confidence", 0.0)) for item in source_entries],
                [
                    max(float(item.get("strength", 0.0)), 0.01)
                    for item in source_entries
                ],
            ) * 0.50

        source_support = self._calculate_source_support(source_entries)
        source_diversity = self._calculate_source_diversity(source_entries)

        global_breakdown = calculate_global_score(
            raw_coverage=raw_coverage,
            weighted_coverage=weighted_coverage,
            effective_coverage=effective_coverage,
            confidence=confidence,
            continuity_top3=float(continuity["top3_run_ratio"]),
            source_support=source_support,
            source_diversity=source_diversity,
        )

        top_sources = [
            {
                "file": item["file"],
                "engine": item["engine"],
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
            "global_score": float(global_breakdown.global_score),
            "global_score_level": global_breakdown.score_level,
            "global_score_raw": float(global_breakdown.raw_score),
            "global_low_evidence_cap": float(global_breakdown.low_evidence_cap),
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
            "global_supported_source_count": int(
                sum(
                    1
                    for item in source_entries
                    if float(item.get("effective_coverage", 0.0)) > 0.0
                )
            ),
            "global_candidate_count": int(candidate_count if candidate_count is not None else len(verified_results)),
            "global_reference_count": int(reference_count if reference_count is not None else len(verified_results)),
            "global_hit_count": int(total_hits),
            "global_interval_hit_count": int(interval_hits),
            "global_target_length": int(target_len),
            "top_sources": top_sources,
        }
