from typing import Dict, Optional, Tuple


DEFAULT_PROFILE = "balanced"

THRESHOLD_PROFILES = {
            "strict": {
                "short_text_max_chars": 200,
                "outlier_std_k": 2.3,
                "outlier_percentile": 0.98,
                "outlier_percentile_margin": 0.012,
                "outlier_percentile_min_windows": 12,
                "short_low": 0.66,
                "short_high": 0.74,
                "long_low": 0.86,
                "long_high": 0.91,
                "paragraph_threshold": 0.97,
                "min_window_chars": 12,
                "score_weights": {
                    "doc_semantic": 0.20,
                    "coverage": 0.30,
                    "confidence": 0.50,
                },
                "final_score": {
                    "semantic_weight": 0.24,
                    "evidence_weight": 0.76,
                    "semantic_center": 0.84,
                    "semantic_scale": 0.07,
                    "coverage_gain": 9.0,
                    "low_evidence_cap_base": 0.10,
                    "low_evidence_cap_gain": 0.16,
                    "continuity_boost": 0.05,
                },
                "semantic_floor": 0.80,
                "score_gate": {
                    "base": 0.08,
                    "coverage": 0.55,
                    "confidence": 0.30,
                    "evidence": 0.25,
                    "low_cov_th": 0.05,
                    "mid_cov_th": 0.08,
                    "mid_conf_th": 0.35,
                    "mid_cov_cap": 0.28,
                    "topic_cov_th": 0.12,
                    "topic_conf_th": 0.75,
                    "topic_cap": 0.20,
                    "low_evidence_cov_th": 0.10,
                    "low_evidence_conf_th": 0.60,
                    "low_evidence_cap": 0.08,
                },
            },
            "balanced": {
                "short_text_max_chars": 200,
                "outlier_std_k": 2.0,
                "outlier_percentile": 0.95,
                "outlier_percentile_margin": 0.010,
                "outlier_percentile_min_windows": 10,
                "short_low": 0.60,
                "short_high": 0.70,
                "long_low": 0.82,
                "long_high": 0.88,
                "paragraph_threshold": 0.95,
                "min_window_chars": 10,
                "score_weights": {
                    "doc_semantic": 0.18,
                    "coverage": 0.52,
                    "confidence": 0.30,
                },
                "final_score": {
                    "semantic_weight": 0.30,
                    "evidence_weight": 0.70,
                    "semantic_center": 0.80,
                    "semantic_scale": 0.08,
                    "coverage_gain": 8.0,
                    "low_evidence_cap_base": 0.14,
                    "low_evidence_cap_gain": 0.20,
                    "continuity_boost": 0.06,
                },
                "semantic_floor": 0.86,
                "score_gate": {
                    "base": 0.09,
                    "coverage": 0.60,
                    "confidence": 0.22,
                    "evidence": 0.22,
                    "low_cov_th": 0.05,
                    "low_cov_cap": 0.20,
                    "mid_cov_th": 0.08,
                    "mid_conf_th": 0.35,
                    "mid_cov_cap": 0.30,
                    "topic_cov_th": 0.15,
                    "topic_conf_th": 0.75,
                    "topic_cap": 0.24,
                    "low_evidence_cov_th": 0.10,
                    "low_evidence_conf_th": 0.55,
                    "low_evidence_cap": 0.14,
                },
            },
            "recall": {
                "short_text_max_chars": 200,
                "outlier_std_k": 1.7,
                "outlier_percentile": 0.93,
                "outlier_percentile_margin": 0.008,
                "outlier_percentile_min_windows": 8,
                "short_low": 0.55,
                "short_high": 0.65,
                "long_low": 0.78,
                "long_high": 0.84,
                "paragraph_threshold": 0.92,
                "min_window_chars": 8,
                "score_weights": {
                    "doc_semantic": 0.35,
                    "coverage": 0.40,
                    "confidence": 0.25,
                },
                "final_score": {
                    "semantic_weight": 0.36,
                    "evidence_weight": 0.64,
                    "semantic_center": 0.76,
                    "semantic_scale": 0.09,
                    "coverage_gain": 7.0,
                    "low_evidence_cap_base": 0.18,
                    "low_evidence_cap_gain": 0.22,
                    "continuity_boost": 0.07,
                },
                "semantic_floor": 0.74,
                "score_gate": {
                    "base": 0.15,
                    "coverage": 0.50,
                    "confidence": 0.25,
                    "evidence": 0.20,
                    "low_cov_th": 0.04,
                    "low_cov_cap": 0.22,
                    "mid_cov_th": 0.07,
                    "mid_conf_th": 0.30,
                    "mid_cov_cap": 0.30,
                    "topic_cov_th": 0.10,
                    "topic_conf_th": 0.70,
                    "topic_cap": 0.26,
                },
            },
        }


def resolve_profile(
    profile_name: Optional[str],
    default_profile: str = DEFAULT_PROFILE,
    threshold_profiles: Dict[str, Dict] = THRESHOLD_PROFILES,
) -> Tuple[str, Dict]:
    if not isinstance(profile_name, str) or not profile_name.strip():
        return default_profile, threshold_profiles[default_profile]

    normalized = profile_name.strip().lower()
    if normalized not in threshold_profiles:
        print(f">>> [BGE][Warn] Unknown threshold profile: {profile_name}. Fallback to {default_profile}.")
        normalized = default_profile
    return normalized, threshold_profiles[normalized]
