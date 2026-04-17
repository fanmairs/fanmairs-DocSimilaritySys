from __future__ import annotations

from typing import Iterable, List, Tuple


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    normalized = sorted(
        (int(start), int(end))
        for start, end in intervals
        if int(end) > int(start)
    )
    if not normalized:
        return []

    merged = [normalized[0]]
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def sum_intervals(intervals: Iterable[Tuple[int, int]]) -> int:
    return int(sum(max(0, int(end) - int(start)) for start, end in intervals))
