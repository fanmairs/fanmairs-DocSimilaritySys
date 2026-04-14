import re
from typing import Dict, Optional

import numpy as np
from document_readers.common.noise_filter import is_numeric_table_noise


def select_topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    if scores.size == 0 or topk <= 0:
        return np.asarray([], dtype=int)

    k = min(int(topk), int(scores.size))
    if k == scores.size:
        return np.argsort(scores)[::-1]

    candidate_idx = np.argpartition(scores, -k)[-k:]
    return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]


def resolve_outlier_metrics(engine, sims: np.ndarray, peak_sim: float, profile_cfg: Dict) -> Dict[str, float]:
    if sims.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "std_threshold": 0.0,
            "percentile_threshold": 0.0,
            "effective_threshold": 0.0,
            "is_statistical_outlier": False,
            "is_percentile_outlier": False,
            "used_percentile": False,
        }

    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))
    outlier_std_k = float(profile_cfg.get("outlier_std_k", 2.0))
    std_threshold = mean_sim + outlier_std_k * std_sim

    percentile_cfg = float(profile_cfg.get("outlier_percentile", 0.0))
    if percentile_cfg > 1.0:
        percentile_cfg /= 100.0
    percentile_cfg = min(max(percentile_cfg, 0.0), 0.999)
    percentile_margin = float(profile_cfg.get("outlier_percentile_margin", 0.0))
    percentile_min_windows = max(1, int(profile_cfg.get("outlier_percentile_min_windows", 1)))

    percentile_threshold = 0.0
    used_percentile = percentile_cfg > 0.0 and sims.size >= percentile_min_windows
    is_percentile_outlier = False
    if used_percentile:
        percentile_threshold = float(np.quantile(sims, percentile_cfg)) + percentile_margin
        is_percentile_outlier = peak_sim >= percentile_threshold

    is_statistical_outlier = peak_sim > std_threshold
    effective_threshold = std_threshold
    if used_percentile:
        effective_threshold = min(std_threshold, percentile_threshold)

    return {
        "mean": mean_sim,
        "std": std_sim,
        "std_threshold": float(std_threshold),
        "percentile_threshold": float(percentile_threshold),
        "effective_threshold": float(effective_threshold),
        "is_statistical_outlier": bool(is_statistical_outlier),
        "is_percentile_outlier": bool(is_percentile_outlier),
        "used_percentile": bool(used_percentile),
    }


def score_window_candidate(
    engine,
    item1: Dict,
    item2: Dict,
    raw_sim: float,
    outlier_threshold: float,
    profile_cfg: Dict,
) -> Optional[Dict]:
    import difflib

    text1 = item1["text"]
    text2 = item2["text"]

    min_window_chars = profile_cfg["min_window_chars"]
    if len(text1) <= min_window_chars or len(text2) <= min_window_chars:
        return None
    if is_numeric_table_noise(text1) or is_numeric_table_noise(text2):
        return None

    digit_ratio1 = sum(c.isdigit() for c in text1) / max(1, len(text1))
    digit_ratio2 = sum(c.isdigit() for c in text2) / max(1, len(text2))
    eng_ratio1 = len(re.findall(r'[a-zA-Z]', text1)) / max(1, len(text1))

    edit_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

    entities1 = engine._extract_entities(text1)
    entities2 = engine._extract_entities(text2)
    entity_iou = engine._safe_iou(entities1, entities2)

    tags1 = engine._extract_tags(text1)
    tags2 = engine._extract_tags(text2)
    tag_iou = engine._safe_iou(tags1, tags2)

    skeleton1 = engine._get_skeleton(text1)
    skeleton2 = engine._get_skeleton(text2)
    skeleton_sim = difflib.SequenceMatcher(None, skeleton1, skeleton2).ratio()

    rule_penalty = 1.0
    rule_flags = []

    if entity_iou < 0.1 and tag_iou < 0.25 and skeleton_sim > 0.8:
        rule_penalty *= 0.55
        rule_flags.append("template_skeleton")

    if engine._is_formula_explanation(text1) and engine._is_formula_explanation(text2):
        if entity_iou > 0.3 and edit_similarity < 0.8:
            rule_penalty *= 0.70
            rule_flags.append("formula_explanation")

    if entity_iou < 0.08:
        rule_penalty *= 0.88
        rule_flags.append("entity_mismatch")
    elif entity_iou < 0.16:
        rule_penalty *= 0.93

    if tag_iou < 0.10:
        rule_penalty *= 0.92
        rule_flags.append("tag_mismatch")

    # Keep technical / formula-heavy spans as downgraded evidence instead of hard rejecting them.
    # This matches the Step5 goal: prefer penalty over silent discard on boundary samples.
    if text1.count('.') >= 5:
        rule_penalty *= 0.92
        rule_flags.append("target_many_periods")
    if text2.count('.') >= 5:
        rule_penalty *= 0.92
        rule_flags.append("ref_many_periods")
    if digit_ratio1 >= 0.2:
        rule_penalty *= 0.84
        rule_flags.append("target_digit_heavy")
    if digit_ratio2 >= 0.2:
        rule_penalty *= 0.84
        rule_flags.append("ref_digit_heavy")
    if eng_ratio1 >= 0.4:
        rule_penalty *= 0.88
        rule_flags.append("target_english_heavy")

    eng_ratio2 = len(re.findall(r'[a-zA-Z]', text2)) / max(1, len(text2))
    if eng_ratio2 >= 0.4:
        rule_penalty *= 0.88
        rule_flags.append("ref_english_heavy")

    margin = raw_sim - outlier_threshold
    margin_norm = engine._clamp01((margin + 0.05) / 0.20)

    confidence = raw_sim * (0.65 + 0.35 * margin_norm)
    confidence *= (0.90 + 0.10 * entity_iou)
    confidence *= (0.90 + 0.10 * tag_iou)
    confidence *= rule_penalty
    if edit_similarity < 0.12:
        confidence *= 0.90

    effective_score = engine._clamp01(raw_sim * (0.55 + 0.45 * rule_penalty))
    return {
        'target_part': text1,
        'ref_part': text2,
        'score': float(effective_score),
        'raw_score': float(raw_sim),
        'confidence': float(engine._clamp01(confidence)),
        'length': len(text1),
        'target_start': int(item1['start']),
        'target_end': int(item1['end']),
        'ref_start': int(item2['start']),
        'ref_end': int(item2['end']),
        'match_type': 'window',
        'rule_penalty': float(engine._clamp01(rule_penalty)),
        'rule_flags': rule_flags,
    }
