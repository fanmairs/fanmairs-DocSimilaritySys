from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scoring.common import clamp01


@dataclass(frozen=True)
class EvidenceSpan:
    target_part: str
    ref_part: str
    score: float
    confidence: float
    length: int
    target_start: Optional[int] = None
    target_end: Optional[int] = None
    ref_start: Optional[int] = None
    ref_end: Optional[int] = None
    match_type: str = "window"
    engine: str = "unknown"
    source: Optional[str] = None
    raw_score: Optional[float] = None
    rule_penalty: Optional[float] = None
    rule_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "target_part": self.target_part,
            "ref_part": self.ref_part,
            "score": clamp01(self.score),
            "confidence": clamp01(self.confidence),
            "length": max(0, int(self.length)),
            "target_start": self.target_start,
            "target_end": self.target_end,
            "ref_start": self.ref_start,
            "ref_end": self.ref_end,
            "match_type": self.match_type,
            "engine": self.engine,
            "source": self.source,
            "rule_flags": list(self.rule_flags),
        }
        if self.raw_score is not None:
            data["raw_score"] = clamp01(self.raw_score)
        if self.rule_penalty is not None:
            data["rule_penalty"] = clamp01(self.rule_penalty)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class EvidenceSummary:
    raw_coverage: float
    weighted_coverage: float
    effective_coverage: float
    confidence: float
    hit_count: int
    interval_hit_count: int
    longest_run_ratio: float
    top3_run_ratio: float
    merged_hit_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_raw": clamp01(self.raw_coverage),
            "coverage_weighted": clamp01(self.weighted_coverage),
            "coverage_effective": clamp01(self.effective_coverage),
            "confidence": clamp01(self.confidence),
            "hit_count": int(self.hit_count),
            "interval_hit_count": int(self.interval_hit_count),
            "continuity_longest": clamp01(self.longest_run_ratio),
            "continuity_top3": clamp01(self.top3_run_ratio),
            "merged_hit_count": int(self.merged_hit_count),
        }
