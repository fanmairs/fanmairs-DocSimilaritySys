from typing import Any, Dict, Optional

from engines.base import EnginePairResult


class SemanticEngine:
    """Adapter around the existing BGE deep semantic engine."""

    kind = "semantic"

    def __init__(
        self,
        engine: Optional[Any] = None,
        model_name: str = "BAAI/bge-large-zh-v1.5",
    ):
        if engine is not None:
            self.engine = engine
        else:
            from .bge_backend import DeepSemanticEngine

            self.engine = DeepSemanticEngine(model_name=model_name)

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def compare_pair(
        self,
        target_text: str,
        reference_text: str,
        threshold_profile: str = "balanced",
        reference: Optional[str] = None,
    ) -> EnginePairResult:
        matches = self.engine.sliding_window_check(
            target_text,
            reference_text,
            threshold_profile=threshold_profile,
        )
        breakdown: Dict[str, Any] = self.engine.score_document_pair(
            target_text,
            reference_text,
            plagiarized_parts=matches,
            threshold_profile=threshold_profile,
        )
        score = float(breakdown.get("final_score", 0.0))
        risk_score = float(breakdown.get("risk_score", score))
        return EnginePairResult(
            score=score,
            risk_score=risk_score,
            matches=matches,
            breakdown=breakdown,
            engine="semantic",
            reference=reference,
        )

    def estimate_window_count(self, text: str) -> int:
        normalized = self.engine._normalize_text(text)
        if not normalized:
            return 0
        return len(self.engine._build_windows(normalized))
