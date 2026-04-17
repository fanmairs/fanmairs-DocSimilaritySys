from __future__ import annotations

from collections import defaultdict
import heapq
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scoring.common import clamp01

from .intervals import merge_intervals, sum_intervals
from .models import EvidenceSummary


def collect_target_intervals(
    evidence_parts: Sequence[Mapping[str, object]],
    target_len: int,
) -> List[Tuple[int, int]]:
    if target_len <= 0 or not evidence_parts:
        return []

    intervals: List[Tuple[int, int]] = []
    for part in evidence_parts:
        start = part.get("target_start")
        end = part.get("target_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            s = max(0, start)
            e = min(target_len, end)
            if e > s:
                intervals.append((s, e))
    return intervals


def collect_interval_parts(evidence_parts: Sequence[Mapping[str, object]]) -> List[Dict]:
    interval_parts: List[Dict] = []
    for part in evidence_parts:
        start = part.get("target_start")
        end = part.get("target_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            interval_parts.append(dict(part))
    return interval_parts


def calculate_raw_coverage(
    evidence_parts: Sequence[Mapping[str, object]],
    target_len: int,
) -> float:
    intervals = collect_target_intervals(evidence_parts, target_len)
    if not intervals:
        return 0.0

    merged = merge_intervals(intervals)
    return clamp01(sum_intervals(merged) / max(1, target_len))


def calculate_coverage(
    evidence_parts: Sequence[Mapping[str, object]],
    target_len: int,
) -> float:
    if target_len <= 0 or not evidence_parts:
        return 0.0

    events = []
    for part in evidence_parts:
        start = part.get("target_start")
        end = part.get("target_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            s = max(0, start)
            e = min(target_len, end)
            if e <= s:
                continue
            conf = clamp01(float(part.get("confidence", part.get("score", 0.0))))
            conf = conf ** 1.10
            events.append((s, 1, round(conf, 6)))
            events.append((e, -1, round(conf, 6)))

    if not events:
        return 0.0

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

    return clamp01(weighted_covered / max(1, target_len))


def calculate_match_confidence(evidence_parts: Sequence[Mapping[str, object]]) -> float:
    if not evidence_parts:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    conf_samples = []
    for part in evidence_parts:
        confidence = clamp01(float(part.get("confidence", part.get("score", 0.0))))
        weight = float(max(1, int(part.get("length", 1))))
        weighted_sum += confidence * weight
        total_weight += weight
        conf_samples.append(confidence)

    if total_weight <= 0:
        return 0.0

    raw_conf = clamp01(weighted_sum / total_weight)
    if conf_samples:
        topk = min(5, len(conf_samples))
        top_mean = float(np.mean(np.sort(np.asarray(conf_samples))[-topk:]))
        raw_conf = clamp01(0.70 * raw_conf + 0.30 * top_mean)

    return clamp01(raw_conf ** 1.35)


def calculate_effective_coverage(
    raw_coverage: float,
    weighted_coverage: float,
    confidence: float,
) -> float:
    raw_coverage = clamp01(raw_coverage)
    weighted_coverage = clamp01(weighted_coverage)
    confidence = clamp01(confidence)

    coverage_gap = max(0.0, raw_coverage - weighted_coverage)
    recovery = coverage_gap * (0.20 + 0.35 * confidence)
    return clamp01(weighted_coverage + recovery)


def calculate_continuity_features(
    evidence_parts: Sequence[Mapping[str, object]],
    target_len: int,
) -> Dict[str, float]:
    intervals = collect_target_intervals(evidence_parts, target_len)
    if not intervals:
        return {
            "longest_run_ratio": 0.0,
            "top3_run_ratio": 0.0,
            "merged_hit_count": 0,
        }

    merged = merge_intervals(intervals)
    lengths = sorted((max(0, end - start) for start, end in merged), reverse=True)
    if not lengths:
        return {
            "longest_run_ratio": 0.0,
            "top3_run_ratio": 0.0,
            "merged_hit_count": 0,
        }

    longest_run_ratio = clamp01(lengths[0] / max(1, target_len))
    top3_run_ratio = clamp01(sum(lengths[:3]) / max(1, target_len))
    return {
        "longest_run_ratio": float(longest_run_ratio),
        "top3_run_ratio": float(top3_run_ratio),
        "merged_hit_count": int(len(merged)),
    }


def summarize_evidence(
    evidence_parts: Sequence[Mapping[str, object]],
    target_len: int,
) -> EvidenceSummary:
    interval_parts = collect_interval_parts(evidence_parts)
    raw_coverage = calculate_raw_coverage(interval_parts, target_len)
    weighted_coverage = calculate_coverage(interval_parts, target_len)
    confidence_input = interval_parts if interval_parts else evidence_parts
    confidence = calculate_match_confidence(confidence_input)
    effective_coverage = calculate_effective_coverage(
        raw_coverage,
        weighted_coverage,
        confidence,
    )
    continuity = calculate_continuity_features(interval_parts, target_len)

    return EvidenceSummary(
        raw_coverage=raw_coverage,
        weighted_coverage=weighted_coverage,
        effective_coverage=effective_coverage,
        confidence=confidence,
        hit_count=len(evidence_parts),
        interval_hit_count=len(interval_parts),
        longest_run_ratio=float(continuity["longest_run_ratio"]),
        top3_run_ratio=float(continuity["top3_run_ratio"]),
        merged_hit_count=int(continuity["merged_hit_count"]),
    )
