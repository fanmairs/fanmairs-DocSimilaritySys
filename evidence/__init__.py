"""Evidence models and aggregation utilities shared by all engines."""

from .adapters import normalize_evidence_span, normalize_evidence_spans
from .global_aggregator import GlobalEvidenceAggregator
from .intervals import merge_intervals, sum_intervals
from .metrics import (
    calculate_continuity_features,
    calculate_coverage,
    calculate_effective_coverage,
    calculate_match_confidence,
    calculate_raw_coverage,
    collect_interval_parts,
    collect_target_intervals,
    summarize_evidence,
)
from .models import EvidenceSpan, EvidenceSummary

__all__ = [
    "EvidenceSpan",
    "EvidenceSummary",
    "GlobalEvidenceAggregator",
    "calculate_continuity_features",
    "calculate_coverage",
    "calculate_effective_coverage",
    "calculate_match_confidence",
    "calculate_raw_coverage",
    "collect_interval_parts",
    "collect_target_intervals",
    "merge_intervals",
    "normalize_evidence_span",
    "normalize_evidence_spans",
    "sum_intervals",
    "summarize_evidence",
]
