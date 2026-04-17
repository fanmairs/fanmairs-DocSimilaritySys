"""Traditional white-box similarity engine package."""

__all__ = [
    "PlagiarismDetectorSystem",
    "SoftSemanticScorer",
    "TraditionalEngine",
    "WhiteBoxLSA",
    "WhiteBoxTFIDF",
    "WindowDetector",
    "calculate_risk_score",
    "fuse_similarity_scores",
]


def __getattr__(name):
    if name == "TraditionalEngine":
        from .engine import TraditionalEngine

        return TraditionalEngine
    if name == "PlagiarismDetectorSystem":
        from .system import PlagiarismDetectorSystem

        return PlagiarismDetectorSystem
    if name == "WhiteBoxTFIDF":
        from .tfidf_backend import WhiteBoxTFIDF

        return WhiteBoxTFIDF
    if name == "WhiteBoxLSA":
        from .lsa_backend import WhiteBoxLSA

        return WhiteBoxLSA
    if name == "WindowDetector":
        from .window_detector import WindowDetector

        return WindowDetector
    if name == "SoftSemanticScorer":
        from .soft_semantic import SoftSemanticScorer

        return SoftSemanticScorer
    if name in {"calculate_risk_score", "fuse_similarity_scores"}:
        from . import scoring

        return getattr(scoring, name)
    raise AttributeError(name)
