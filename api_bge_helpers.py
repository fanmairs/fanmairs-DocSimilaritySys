import json
import os
from typing import Dict, List, Optional

from fastapi import HTTPException

from coarse_retrieval import CoarseRetrievalConfig
from deep_semantic import DeepSemanticEngine


BGE_STRATEGY_COARSE = "coarse_then_fine"
BGE_STRATEGY_FULL = "full_fine"
BGE_STRATEGIES = {BGE_STRATEGY_COARSE, BGE_STRATEGY_FULL}


def resolve_bge_strategy(value: Optional[str]) -> str:
    normalized = (value or BGE_STRATEGY_COARSE).strip().lower()
    return normalized if normalized in BGE_STRATEGIES else BGE_STRATEGY_COARSE


def parse_coarse_config_payload(payload: Optional[str]) -> Optional[Dict[str, object]]:
    if payload is None:
        return None

    normalized = payload.strip()
    if not normalized:
        return None

    try:
        raw_config = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="coarse_config 必须是合法 JSON") from exc

    if not isinstance(raw_config, dict):
        raise HTTPException(status_code=400, detail="coarse_config 必须是 JSON 对象")

    try:
        validated = CoarseRetrievalConfig.from_partial_dict(raw_config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return validated.to_dict()


def run_bert_fine_verification(bert_engine, ref_path: str, target_text: str, ref_text: str, bert_profile: str):
    plagiarized_parts = bert_engine.sliding_window_check(
        target_text,
        ref_text,
        threshold_profile=bert_profile,
    )
    score_breakdown = bert_engine.score_document_pair(
        target_text,
        ref_text,
        plagiarized_parts=plagiarized_parts,
        threshold_profile=bert_profile,
    )
    print(
        ">>> [BGE][Score] "
        f"file={os.path.basename(ref_path)} "
        "stage=fine "
        f"final={score_breakdown['final_score']:.4f} "
        f"risk={score_breakdown.get('risk_score', score_breakdown['final_score']):.4f} "
        f"doc={score_breakdown['doc_semantic']:.4f} "
        f"doc_ex={score_breakdown.get('doc_semantic_excess', score_breakdown['doc_semantic']):.4f} "
        f"cov={score_breakdown['coverage']:.4f} "
        f"cov_eff={score_breakdown.get('coverage_effective', score_breakdown['coverage']):.4f} "
        f"cov_w={score_breakdown.get('coverage_weighted', score_breakdown['coverage']):.4f} "
        f"conf={score_breakdown['confidence']:.4f} "
        f"base={score_breakdown['base_score']:.4f} "
        f"gate={score_breakdown['gate']:.4f} "
        f"hits={score_breakdown['hit_count']}"
    )
    return plagiarized_parts, score_breakdown


def build_basic_bert_result(
    ref_path: str,
    bert_profile: str,
    score_breakdown: Dict[str, float],
    plagiarized_parts: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "file": os.path.basename(ref_path).replace("ref_", ""),
        "sim_bert": float(score_breakdown["final_score"]),
        "sim_bert_risk": float(score_breakdown.get("risk_score", score_breakdown["final_score"])),
        "sim_bert_doc": float(score_breakdown["doc_semantic"]),
        "sim_bert_doc_excess": float(score_breakdown.get("doc_semantic_excess", score_breakdown["doc_semantic"])),
        "sim_bert_coverage": float(score_breakdown["coverage"]),
        "sim_bert_coverage_raw": float(score_breakdown.get("coverage_raw", score_breakdown["coverage"])),
        "sim_bert_coverage_weighted": float(score_breakdown.get("coverage_weighted", score_breakdown["coverage"])),
        "sim_bert_coverage_effective": float(score_breakdown.get("coverage_effective", score_breakdown["coverage"])),
        "sim_bert_confidence": float(score_breakdown["confidence"]),
        "sim_bert_base": float(score_breakdown["base_score"]),
        "sim_bert_gate": float(score_breakdown["gate"]),
        "sim_bert_hits": int(score_breakdown["hit_count"]),
        "sim_bert_semantic_signal": float(score_breakdown.get("semantic_signal", 0.0)),
        "sim_bert_evidence": float(score_breakdown.get("evidence_score", 0.0)),
        "sim_bert_continuity_bonus": float(score_breakdown.get("continuity_bonus", 0.0)),
        "sim_bert_continuity_longest": float(score_breakdown.get("continuity_longest", 0.0)),
        "sim_bert_continuity_top3": float(score_breakdown.get("continuity_top3", 0.0)),
        "sim_bert_low_evidence_cap": float(score_breakdown.get("low_evidence_cap", 1.0)),
        "sim_bert_legacy_coverage": float(score_breakdown.get("coverage_raw", score_breakdown["coverage"])),
        "sim_bert_verified": True,
        "sim_bert_candidate": True,
        "bert_profile": bert_profile,
        "retrieval_stage": "fine_verified",
        "plagiarized_parts": plagiarized_parts,
    }


def estimate_text_window_count(bert_engine, text: str) -> int:
    normalized = DeepSemanticEngine._normalize_text(text)
    if not normalized:
        return 0
    return len(bert_engine._build_windows(normalized))


def window_scale_level(pair_count: int) -> str:
    if pair_count >= 50000:
        return "large"
    if pair_count >= 12000:
        return "medium"
    return "small"


def window_recommendation(pair_count: int, reference_count: int) -> Dict[str, str]:
    scale_level = window_scale_level(pair_count)
    if scale_level == "small" and reference_count <= 12:
        return {
            "strategy": BGE_STRATEGY_FULL,
            "label": "建议完整细检",
            "message": "当前窗口规模较小，完整细检的等待成本可控，更适合追求结果完整性。",
        }
    if scale_level == "medium":
        return {
            "strategy": BGE_STRATEGY_COARSE,
            "label": "建议粗筛后细检",
            "message": "当前窗口规模已经明显上升，粗筛后细检可以降低等待时间。",
        }
    if scale_level == "large":
        return {
            "strategy": BGE_STRATEGY_COARSE,
            "label": "强烈建议粗筛后细检",
            "message": "当前全量细检矩阵较大，完整模式可能等待较久，建议先粗筛候选。",
        }
    return {
        "strategy": BGE_STRATEGY_COARSE,
        "label": "建议粗筛后细检",
        "message": "参考文档数量较多，建议用粗筛保留可疑来源，再进入细粒度复核。",
    }
