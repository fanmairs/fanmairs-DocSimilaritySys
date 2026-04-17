"""Similarity engine adapters and factories."""

from .base import EngineKind, EnginePairResult, UnsupportedEngineError
from .factory import create_engine, resolve_engine_kind

__all__ = [
    "EngineKind",
    "EnginePairResult",
    "UnsupportedEngineError",
    "create_engine",
    "resolve_engine_kind",
]
