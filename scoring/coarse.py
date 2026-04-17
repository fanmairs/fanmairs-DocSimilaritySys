from __future__ import annotations

from typing import Sequence

import numpy as np

from .common import clamp01


def compose_coarse_score(
    *,
    doc_semantic: float,
    paragraph_hotspot: float,
    lexical_anchor: float,
) -> float:
    """Blend coarse semantic and lexical signals before expensive verification."""
    doc_semantic = clamp01(doc_semantic)
    paragraph_hotspot = clamp01(paragraph_hotspot)
    lexical_anchor = clamp01(lexical_anchor)

    score = 0.45 * paragraph_hotspot + 0.35 * doc_semantic + 0.20 * lexical_anchor

    # Same-topic documents can be semantically close without concrete overlap.
    if doc_semantic >= 0.86 and paragraph_hotspot < 0.58 and lexical_anchor < 0.12:
        score = min(score, 0.62)

    if paragraph_hotspot >= 0.82:
        score = max(score, 0.55 * paragraph_hotspot + 0.25 * doc_semantic + 0.20 * lexical_anchor)

    return clamp01(score)


def calculate_paragraph_hotspot(
    best_per_target: Sequence[float] | np.ndarray,
    *,
    top_k: int,
) -> float:
    """Summarize target-paragraph best matches into one coarse hotspot score."""
    scores = np.asarray(best_per_target, dtype=float)
    if scores.size == 0:
        return 0.0

    k = min(max(1, int(top_k)), int(scores.size))
    topk_mean = float(np.mean(np.sort(scores)[-k:]))
    overall_mean = float(np.mean(scores))
    return clamp01(0.70 * topk_mean + 0.30 * overall_mean)
