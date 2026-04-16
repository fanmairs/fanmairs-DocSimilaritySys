"""Compatibility wrapper for the BGE semantic engine.

New code should import from ``engines.semantic.bge_backend``.
"""

from engines.semantic.bge_backend import DeepSemanticEngine

__all__ = ["DeepSemanticEngine"]
