def fuse_similarity_scores(
    sim_lsa,
    sim_tfidf,
    sim_soft,
    target_token_len=0,
    ref_token_len=0,
    semantic_weight=0.35,
):
    """
    Blend lexical and semantic signals for traditional mode.
    Optimization goals:
    1) Keep LSA semantic power.
    2) Suppress LSA-only spikes when lexical/soft evidence is weak.
    3) Reduce score bias when document lengths are extremely unbalanced.
    """
    lexical_support = max(sim_tfidf, sim_soft)

    # Dynamic weighting by evidence strength.
    if lexical_support < 0.05:
        w_lsa, w_tfidf, w_soft = 0.32, 0.53, 0.15
    elif lexical_support < 0.12:
        w_lsa, w_tfidf, w_soft = 0.44, 0.43, 0.13
    elif lexical_support < 0.25:
        w_lsa, w_tfidf, w_soft = 0.56, 0.31, 0.13
    else:
        w_lsa, w_tfidf, w_soft = 0.60, 0.24, 0.16

    score = w_lsa * sim_lsa + w_tfidf * sim_tfidf + w_soft * sim_soft

    # Preserve configurable semantic boost while keeping it bounded.
    score += semantic_weight * 0.30 * sim_soft

    # Penalize "LSA dominates everything else" to reduce false positives.
    lsa_gap = sim_lsa - lexical_support
    if lsa_gap > 0.35:
        score -= min(0.30, (lsa_gap - 0.35) * 0.55)

    # Length balance penalty for very asymmetric documents.
    if target_token_len > 0 and ref_token_len > 0:
        ratio = min(target_token_len, ref_token_len) / max(target_token_len, ref_token_len)
        if ratio < 0.2 and lexical_support < 0.12:
            score *= 0.88
        elif ratio < 0.1 and lexical_support < 0.08:
            score *= 0.80

    return max(0.0, min(score, 1.0))


def calculate_risk_score(sim_hybrid, sim_lsa, sim_tfidf, sim_soft):
    """
    Risk score is used for alarm level only.
    It prevents false negatives when Hybrid is intentionally conservative.
    """
    lexical_anchor = 0.72 * sim_tfidf + 0.28 * sim_lsa
    semantic_anchor = 0.60 * sim_lsa + 0.40 * sim_soft
    return max(sim_hybrid, lexical_anchor, semantic_anchor)
