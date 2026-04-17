"""Compatibility layer for traditional score fusion.

New code should import from ``scoring.traditional``.
"""

from scoring.traditional import calculate_risk_score, fuse_similarity_scores

__all__ = ["calculate_risk_score", "fuse_similarity_scores"]
