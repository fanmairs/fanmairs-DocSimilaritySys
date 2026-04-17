from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol


EngineKind = Literal["traditional", "semantic"]


class UnsupportedEngineError(ValueError):
    """Raised when no similarity engine is registered for a requested name."""


@dataclass
class EnginePairResult:
    """Common pair-comparison result used by engine adapters."""

    score: float
    risk_score: float
    matches: List[Dict[str, Any]]
    breakdown: Dict[str, Any]
    engine: EngineKind
    reference: Optional[str] = None


class FileSimilarityEngine(Protocol):
    def compare_files(self, target_file: str, reference_files: List[str], **kwargs: Any) -> Any:
        ...


class PairSimilarityEngine(Protocol):
    def compare_pair(self, target_text: str, reference_text: str, **kwargs: Any) -> EnginePairResult:
        ...
