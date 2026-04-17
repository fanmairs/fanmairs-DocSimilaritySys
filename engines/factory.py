from typing import Any

from .base import EngineKind, UnsupportedEngineError


_TRADITIONAL_ALIASES = {
    "traditional",
    "whitebox",
    "white_box",
    "tfidf",
    "lsa",
}
_SEMANTIC_ALIASES = {
    "semantic",
    "deep",
    "deep_semantic",
    "bge",
    "bert",
}


def resolve_engine_kind(name: str) -> EngineKind:
    """Normalize UI/API engine names to internal engine kinds."""
    normalized = (name or "").strip().lower().replace("-", "_")
    if normalized in _TRADITIONAL_ALIASES:
        return "traditional"
    if normalized in _SEMANTIC_ALIASES:
        return "semantic"
    supported = sorted(_TRADITIONAL_ALIASES | _SEMANTIC_ALIASES)
    raise UnsupportedEngineError(
        f"Unsupported engine: {name!r}. Supported aliases: {', '.join(supported)}"
    )


def create_engine(name: str, **kwargs: Any):
    """Create an engine adapter by name.

    Heavy dependencies are imported lazily inside each adapter so simply
    importing this factory stays cheap.
    """
    kind = resolve_engine_kind(name)
    if kind == "traditional":
        from .traditional.engine import TraditionalEngine

        return TraditionalEngine(**kwargs)

    from .semantic.engine import SemanticEngine

    return SemanticEngine(**kwargs)
