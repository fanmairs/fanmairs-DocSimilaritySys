from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from scoring.common import clamp01

from .models import EvidenceSpan


_STANDARD_KEYS = {
    "target_part",
    "ref_part",
    "reference_part",
    "score",
    "confidence",
    "length",
    "target_start",
    "target_end",
    "ref_start",
    "ref_end",
    "reference_start",
    "reference_end",
    "match_type",
    "engine",
    "source",
    "raw_score",
    "rule_penalty",
    "rule_flags",
}


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_evidence_span(
    raw_span: Mapping[str, Any],
    *,
    engine: str,
    source: Optional[str] = None,
    default_match_type: str = "window",
) -> Dict[str, Any]:
    target_part = str(raw_span.get("target_part", "") or "")
    ref_part = str(
        raw_span.get("ref_part", raw_span.get("reference_part", "")) or ""
    )
    score = clamp01(_safe_float(raw_span.get("score", raw_span.get("raw_score", 0.0))))
    confidence = clamp01(_safe_float(raw_span.get("confidence", score), score))
    length = _optional_int(raw_span.get("length"))
    if length is None:
        length = len(target_part)

    raw_flags = raw_span.get("rule_flags", [])
    if isinstance(raw_flags, str):
        rule_flags = [raw_flags]
    elif isinstance(raw_flags, Iterable):
        rule_flags = [str(item) for item in raw_flags]
    else:
        rule_flags = []

    metadata = {
        key: value
        for key, value in raw_span.items()
        if key not in _STANDARD_KEYS
    }

    span = EvidenceSpan(
        target_part=target_part,
        ref_part=ref_part,
        score=score,
        confidence=confidence,
        length=max(0, int(length)),
        target_start=_optional_int(raw_span.get("target_start")),
        target_end=_optional_int(raw_span.get("target_end")),
        ref_start=_optional_int(
            raw_span.get("ref_start", raw_span.get("reference_start"))
        ),
        ref_end=_optional_int(raw_span.get("ref_end", raw_span.get("reference_end"))),
        match_type=str(raw_span.get("match_type", default_match_type) or default_match_type),
        engine=str(raw_span.get("engine", engine) or engine),
        source=raw_span.get("source", source),
        raw_score=(
            clamp01(_safe_float(raw_span.get("raw_score")))
            if raw_span.get("raw_score") is not None
            else None
        ),
        rule_penalty=(
            clamp01(_safe_float(raw_span.get("rule_penalty")))
            if raw_span.get("rule_penalty") is not None
            else None
        ),
        rule_flags=rule_flags,
        metadata=metadata,
    )
    normalized = span.to_dict()

    for key in ("score_tfidf", "score_soft"):
        if key in raw_span:
            normalized[key] = raw_span[key]
    return normalized


def normalize_evidence_spans(
    spans: Iterable[Mapping[str, Any]],
    *,
    engine: str,
    source: Optional[str] = None,
    default_match_type: str = "window",
) -> List[Dict[str, Any]]:
    return [
        normalize_evidence_span(
            span,
            engine=engine,
            source=source,
            default_match_type=default_match_type,
        )
        for span in spans
        if isinstance(span, Mapping)
    ]
