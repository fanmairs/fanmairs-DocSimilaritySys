"""Compatibility wrapper for traditional score fusion."""

from engines.traditional.scoring import calculate_risk_score, fuse_similarity_scores

__all__ = ["calculate_risk_score", "fuse_similarity_scores"]
