from __future__ import annotations

from .common import clamp01


def fuse_similarity_scores(
    sim_lsa: float,
    sim_tfidf: float,
    sim_soft: float,
    target_token_len: int = 0,
    ref_token_len: int = 0,
    semantic_weight: float = 0.35,
) -> float:
    """
    Blend lexical and semantic signals for traditional mode.

    The fusion deliberately suppresses LSA-only spikes when lexical and soft
    semantic support are weak, while still keeping LSA useful for paraphrases.
    """
    sim_lsa = clamp01(sim_lsa)
    sim_tfidf = clamp01(sim_tfidf)
    sim_soft = clamp01(sim_soft)
    semantic_weight = max(0.0, min(float(semantic_weight), 0.6))
    lexical_support = max(sim_tfidf, sim_soft)

    if lexical_support < 0.05:
        w_lsa, w_tfidf, w_soft = 0.32, 0.53, 0.15
    elif lexical_support < 0.12:
        w_lsa, w_tfidf, w_soft = 0.44, 0.43, 0.13
    elif lexical_support < 0.25:
        w_lsa, w_tfidf, w_soft = 0.56, 0.31, 0.13
    else:
        w_lsa, w_tfidf, w_soft = 0.60, 0.24, 0.16

    score = w_lsa * sim_lsa + w_tfidf * sim_tfidf + w_soft * sim_soft
    score += semantic_weight * 0.30 * sim_soft

    lsa_gap = sim_lsa - lexical_support
    if lsa_gap > 0.35:
        score -= min(0.30, (lsa_gap - 0.35) * 0.55)

    if target_token_len > 0 and ref_token_len > 0:
        ratio = min(target_token_len, ref_token_len) / max(target_token_len, ref_token_len)
        if ratio < 0.2 and lexical_support < 0.12:
            score *= 0.88
        elif ratio < 0.1 and lexical_support < 0.08:
            score *= 0.80

    return clamp01(score)


def calculate_risk_score(
    sim_hybrid: float,
    sim_lsa: float,
    sim_tfidf: float,
    sim_soft: float,
) -> float:
    lexical_anchor = 0.72 * clamp01(sim_tfidf) + 0.28 * clamp01(sim_lsa)
    semantic_anchor = 0.60 * clamp01(sim_lsa) + 0.40 * clamp01(sim_soft)
    return clamp01(max(sim_hybrid, lexical_anchor, semantic_anchor))
