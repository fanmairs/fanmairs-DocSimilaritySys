import math
from typing import Dict, List, Tuple

import numpy as np


def collect_target_intervals(engine, plagiarized_parts: List[Dict], target_len: int) -> List[Tuple[int, int]]:
    if target_len <= 0 or not plagiarized_parts:
        return []

    intervals = []
    for part in plagiarized_parts:
        start = part.get('target_start')
        end = part.get('target_end')
        if isinstance(start, int) and isinstance(end, int) and end > start:
            s = max(0, start)
            e = min(target_len, end)
            if e > s:
                intervals.append((s, e))
    return intervals


def calculate_raw_coverage(engine, plagiarized_parts: List[Dict], target_len: int) -> float:
    intervals = engine._collect_target_intervals(plagiarized_parts, target_len)
    if not intervals:
        return 0.0

    merged = engine._merge_intervals(intervals)
    return engine._clamp01(engine._sum_intervals(merged) / max(1, target_len))


def calculate_coverage(engine, plagiarized_parts: List[Dict], target_len: int) -> float:
    if target_len <= 0 or not plagiarized_parts:
        return 0.0

    # Step5: coverage is confidence-weighted interval union,
    # so downgraded matches contribute less instead of hard removal.
    events = []
    for part in plagiarized_parts:
        start = part.get('target_start')
        end = part.get('target_end')
        if isinstance(start, int) and isinstance(end, int) and end > start:
            s = max(0, start)
            e = min(target_len, end)
            if e <= s:
                continue
            conf = engine._clamp01(float(part.get('confidence', part.get('score', 0.0))))
            # Keep very-low confidence hits lower impact, but avoid over-suppressing balanced mode.
            conf = conf ** 1.10
            events.append((s, 1, round(conf, 6)))
            events.append((e, -1, round(conf, 6)))

    if not events:
        return 0.0

    import heapq
    from collections import defaultdict

    events.sort(key=lambda x: x[0])
    active_counts = defaultdict(int)
    max_heap = []

    weighted_covered = 0.0
    idx = 0
    n = len(events)

    while idx < n:
        pos = events[idx][0]
        while idx < n and events[idx][0] == pos:
            _, typ, conf = events[idx]
            if typ == 1:
                active_counts[conf] += 1
                heapq.heappush(max_heap, -conf)
            else:
                active_counts[conf] -= 1
            idx += 1

        while max_heap and active_counts[-max_heap[0]] <= 0:
            heapq.heappop(max_heap)

        if idx >= n:
            break
        next_pos = events[idx][0]
        if next_pos <= pos:
            continue

        if max_heap:
            top_conf = -max_heap[0]
            weighted_covered += (next_pos - pos) * top_conf

    return engine._clamp01(weighted_covered / max(1, target_len))


def calculate_match_confidence(engine, plagiarized_parts: List[Dict]) -> float:
    if not plagiarized_parts:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    conf_samples = []
    for part in plagiarized_parts:
        confidence = engine._clamp01(float(part.get('confidence', part.get('score', 0.0))))
        weight = float(max(1, int(part.get('length', 1))))
        weighted_sum += confidence * weight
        total_weight += weight
        conf_samples.append(confidence)

    if total_weight <= 0:
        return 0.0

    # Blend weighted mean with top-k confidence to avoid "many weak hits dilute all evidence".
    raw_conf = engine._clamp01(weighted_sum / total_weight)
    if conf_samples:
        topk = min(5, len(conf_samples))
        top_mean = float(np.mean(np.sort(np.asarray(conf_samples))[-topk:]))
        raw_conf = engine._clamp01(0.70 * raw_conf + 0.30 * top_mean)

    # Keep some nonlinearity for robustness, but lighter than before.
    return engine._clamp01(raw_conf ** 1.35)


def calculate_effective_coverage(
    engine,
    raw_coverage: float,
    weighted_coverage: float,
    confidence: float,
) -> float:
    raw_coverage = engine._clamp01(raw_coverage)
    weighted_coverage = engine._clamp01(weighted_coverage)
    confidence = engine._clamp01(confidence)

    coverage_gap = max(0.0, raw_coverage - weighted_coverage)
    recovery = coverage_gap * (0.20 + 0.35 * confidence)
    return engine._clamp01(weighted_coverage + recovery)


def calculate_continuity_features(
    engine,
    plagiarized_parts: List[Dict],
    target_len: int,
) -> Dict[str, float]:
    intervals = engine._collect_target_intervals(plagiarized_parts, target_len)
    if not intervals:
        return {
            "longest_run_ratio": 0.0,
            "top3_run_ratio": 0.0,
            "merged_hit_count": 0,
        }

    merged = engine._merge_intervals(intervals)
    lengths = sorted((max(0, end - start) for start, end in merged), reverse=True)
    if not lengths:
        return {
            "longest_run_ratio": 0.0,
            "top3_run_ratio": 0.0,
            "merged_hit_count": 0,
        }

    longest_run_ratio = engine._clamp01(lengths[0] / max(1, target_len))
    top3_run_ratio = engine._clamp01(sum(lengths[:3]) / max(1, target_len))
    return {
        "longest_run_ratio": float(longest_run_ratio),
        "top3_run_ratio": float(top3_run_ratio),
        "merged_hit_count": int(len(merged)),
    }


def calculate_realistic_score(
    engine,
    profile_name: str,
    profile_cfg: Dict,
    raw_coverage: float,
    weighted_coverage: float,
    confidence: float,
    doc_semantic: float,
    paragraph_semantic: float,
    hit_count: int,
    longest_run_ratio: float,
    top3_run_ratio: float,
) -> Tuple[float, float, float, float, float, float]:
    effective_coverage = engine._calculate_effective_coverage(
        raw_coverage,
        weighted_coverage,
        confidence,
    )

    final_cfg = profile_cfg.get("final_score", {})
    semantic_weight = float(final_cfg.get("semantic_weight", 0.30))
    evidence_weight = float(final_cfg.get("evidence_weight", 0.70))
    semantic_center = float(final_cfg.get("semantic_center", 0.80))
    semantic_scale = max(1e-6, float(final_cfg.get("semantic_scale", 0.08)))
    coverage_gain = max(1e-6, float(final_cfg.get("coverage_gain", 8.0)))
    low_evidence_cap_base = float(final_cfg.get("low_evidence_cap_base", 0.14))
    low_evidence_cap_gain = float(final_cfg.get("low_evidence_cap_gain", 0.20))
    continuity_boost = float(final_cfg.get("continuity_boost", 0.06))

    semantic_input = engine._clamp01(0.60 * doc_semantic + 0.40 * paragraph_semantic)
    semantic_score = engine._sigmoid((semantic_input - semantic_center) / semantic_scale)

    coverage_core = engine._clamp01(
        0.55 * effective_coverage
        + 0.25 * longest_run_ratio
        + 0.20 * top3_run_ratio
    )
    evidence_score = engine._clamp01(1.0 - math.exp(-coverage_gain * coverage_core))
    confidence_scale = 0.72 + 0.28 * (engine._clamp01(confidence) ** 0.85)
    evidence_score = engine._clamp01(evidence_score * confidence_scale)

    continuity_signal = engine._clamp01(
        max(longest_run_ratio * 10.0, top3_run_ratio * 5.0)
    )
    continuity_bonus = continuity_boost * continuity_signal * engine._clamp01(confidence)

    final_score = (
        semantic_weight * semantic_score
        + evidence_weight * evidence_score
        + continuity_bonus
    )

    if effective_coverage < 0.03 and longest_run_ratio < 0.01:
        low_evidence_cap = low_evidence_cap_base + low_evidence_cap_gain * semantic_score
        final_score = min(final_score, low_evidence_cap)
    else:
        low_evidence_cap = 1.0

    return (
        engine._clamp01(final_score),
        float(effective_coverage),
        float(semantic_score),
        float(evidence_score),
        float(continuity_bonus),
        float(low_evidence_cap),
    )
