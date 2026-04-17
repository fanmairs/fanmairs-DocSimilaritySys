from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _display_name(path_or_name: object) -> str:
    name = os.path.basename(str(path_or_name or "unknown"))
    return name.replace("ref_", "")


def build_semantic_result(
    ref_path: str,
    bert_profile: str,
    score_breakdown: Mapping[str, object],
    plagiarized_parts: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    final_score = _safe_float(score_breakdown["final_score"])
    risk_score = _safe_float(score_breakdown.get("risk_score", final_score))
    coverage = _safe_float(score_breakdown["coverage"])
    coverage_raw = _safe_float(score_breakdown.get("coverage_raw", coverage))
    coverage_weighted = _safe_float(score_breakdown.get("coverage_weighted", coverage))
    coverage_effective = _safe_float(score_breakdown.get("coverage_effective", coverage))
    doc_semantic = _safe_float(score_breakdown["doc_semantic"])

    return {
        "file": _display_name(ref_path),
        "engine": "semantic",
        "sim_bert": final_score,
        "sim_bert_risk": risk_score,
        "sim_bert_doc": doc_semantic,
        "sim_bert_doc_excess": _safe_float(
            score_breakdown.get("doc_semantic_excess", doc_semantic)
        ),
        "sim_bert_coverage": coverage,
        "sim_bert_coverage_raw": coverage_raw,
        "sim_bert_coverage_weighted": coverage_weighted,
        "sim_bert_coverage_effective": coverage_effective,
        "sim_bert_confidence": _safe_float(score_breakdown["confidence"]),
        "sim_bert_base": _safe_float(score_breakdown["base_score"]),
        "sim_bert_gate": _safe_float(score_breakdown["gate"]),
        "sim_bert_hits": _safe_int(score_breakdown["hit_count"]),
        "sim_bert_semantic_signal": _safe_float(score_breakdown.get("semantic_signal", 0.0)),
        "sim_bert_evidence": _safe_float(score_breakdown.get("evidence_score", 0.0)),
        "sim_bert_continuity_bonus": _safe_float(
            score_breakdown.get("continuity_bonus", 0.0)
        ),
        "sim_bert_continuity_longest": _safe_float(
            score_breakdown.get("continuity_longest", 0.0)
        ),
        "sim_bert_continuity_top3": _safe_float(
            score_breakdown.get("continuity_top3", 0.0)
        ),
        "sim_bert_low_evidence_cap": _safe_float(
            score_breakdown.get("low_evidence_cap", 1.0)
        ),
        "sim_bert_legacy_coverage": coverage_raw,
        "sim_bert_verified": True,
        "sim_bert_candidate": True,
        "bert_profile": bert_profile,
        "retrieval_stage": "fine_verified",
        "plagiarized_parts": list(plagiarized_parts or []),
    }


def build_semantic_coarse_result(
    item: Mapping[str, object],
    bert_profile: str,
) -> Dict[str, object]:
    coarse_score = _safe_float(item.get("coarse_score", 0.0))
    doc_semantic = _safe_float(item.get("doc_semantic", 0.0))
    paragraph_hotspot = _safe_float(item.get("paragraph_hotspot", 0.0))
    lexical_anchor = _safe_float(item.get("lexical_anchor", 0.0))

    return {
        "file": _display_name(item.get("file", "unknown")),
        "engine": "semantic",
        "sim_bert": coarse_score,
        "sim_bert_risk": coarse_score,
        "sim_bert_doc": doc_semantic,
        "sim_bert_doc_excess": doc_semantic,
        "sim_bert_coverage": 0.0,
        "sim_bert_coverage_raw": 0.0,
        "sim_bert_coverage_weighted": 0.0,
        "sim_bert_coverage_effective": 0.0,
        "sim_bert_confidence": 0.0,
        "sim_bert_base": coarse_score,
        "sim_bert_gate": 1.0,
        "sim_bert_hits": 0,
        "sim_bert_semantic_signal": doc_semantic,
        "sim_bert_evidence": paragraph_hotspot,
        "sim_bert_continuity_bonus": 0.0,
        "sim_bert_continuity_longest": 0.0,
        "sim_bert_continuity_top3": 0.0,
        "sim_bert_low_evidence_cap": 1.0,
        "sim_bert_legacy_coverage": 0.0,
        "sim_bert_coarse": coarse_score,
        "sim_bert_coarse_doc": doc_semantic,
        "sim_bert_coarse_para": paragraph_hotspot,
        "sim_bert_coarse_lex": lexical_anchor,
        "sim_bert_verified": False,
        "sim_bert_candidate": bool(item.get("is_candidate", False)),
        "sim_bert_candidate_rank": item.get("candidate_rank"),
        "sim_bert_coarse_rank": item.get("coarse_rank"),
        "retrieval_stage": "coarse_only",
        "retrieval_reason": item.get("candidate_reason", ""),
        "retrieval_candidate_pool_size": _safe_int(item.get("candidate_pool_size", 0)),
        "retrieval_reference_count": _safe_int(item.get("reference_count", 0)),
        "retrieval_theme_mean": _safe_float(item.get("theme_mean", 0.0)),
        "retrieval_theme_std": _safe_float(item.get("theme_std", 0.0)),
        "retrieval_topic_concentrated": bool(item.get("topic_concentrated", False)),
        "bert_profile": bert_profile,
        "plagiarized_parts": [],
    }


def build_semantic_verified_result(
    item: Mapping[str, object],
    bert_profile: str,
    score_breakdown: Mapping[str, object],
    plagiarized_parts: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    result = build_semantic_coarse_result(item, bert_profile)
    result.update(
        build_semantic_result(
            str(item.get("file", "unknown")),
            bert_profile,
            score_breakdown,
            plagiarized_parts,
        )
    )
    result.update(
        {
            "sim_bert_candidate": bool(item.get("is_candidate", True)),
            "sim_bert_candidate_rank": item.get("candidate_rank"),
            "sim_bert_coarse_rank": item.get("coarse_rank"),
            "retrieval_reason": item.get("candidate_reason", ""),
            "retrieval_candidate_pool_size": _safe_int(item.get("candidate_pool_size", 0)),
            "retrieval_reference_count": _safe_int(item.get("reference_count", 0)),
            "retrieval_theme_mean": _safe_float(item.get("theme_mean", 0.0)),
            "retrieval_theme_std": _safe_float(item.get("theme_std", 0.0)),
            "retrieval_topic_concentrated": bool(item.get("topic_concentrated", False)),
        }
    )
    return result


def build_traditional_result(raw_result: Mapping[str, object]) -> Dict[str, object]:
    sim_lsa = _safe_float(raw_result.get("sim_lsa", 0.0))
    sim_hybrid = _safe_float(raw_result.get("sim_hybrid", sim_lsa))
    risk_score = _safe_float(raw_result.get("risk_score", sim_hybrid))

    return {
        "file": _display_name(raw_result.get("file", "unknown")),
        "engine": "traditional",
        "sim_lsa": sim_lsa,
        "sim_tfidf": _safe_float(raw_result.get("sim_tfidf", 0.0)),
        "sim_soft": _safe_float(raw_result.get("sim_soft", 0.0)),
        "sim_hybrid": sim_hybrid,
        "risk_score": risk_score,
        "plagiarized_parts": list(raw_result.get("plagiarized_parts", []) or []),
    }


def sort_report_items(
    items: Iterable[Mapping[str, object]],
    *,
    mode: str,
) -> List[Dict[str, object]]:
    normalized = [dict(item) for item in items]
    if mode == "bert":
        normalized.sort(key=lambda item: _safe_float(item.get("sim_bert", 0.0)), reverse=True)
    else:
        normalized.sort(
            key=lambda item: _safe_float(
                item.get("risk_score", item.get("sim_hybrid", item.get("sim_lsa", 0.0)))
            ),
            reverse=True,
        )
    return normalized


def build_report_payload(
    items: Sequence[Mapping[str, object]],
    summary: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "items": [dict(item) for item in items],
        "summary": dict(summary) if summary is not None else None,
    }
