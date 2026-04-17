"""BGE/deep-semantic similarity engine package."""

__all__ = [
    "CoarseRetriever",
    "CoarseRetrievalConfig",
    "DeepSemanticEngine",
    "GlobalEvidenceAggregator",
    "SemanticEngine",
]


def __getattr__(name):
    if name == "SemanticEngine":
        from .engine import SemanticEngine

        return SemanticEngine
    if name == "DeepSemanticEngine":
        from .bge_backend import DeepSemanticEngine

        return DeepSemanticEngine
    if name in {"CoarseRetriever", "CoarseRetrievalConfig"}:
        from . import coarse_retrieval

        return getattr(coarse_retrieval, name)
    if name == "GlobalEvidenceAggregator":
        from evidence import GlobalEvidenceAggregator

        return GlobalEvidenceAggregator
    raise AttributeError(name)
