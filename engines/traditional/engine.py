from typing import Any, List, Optional

from config.settings import get_settings


class TraditionalEngine:
    """Adapter around the existing white-box plagiarism system."""

    kind = "traditional"

    def __init__(
        self,
        system: Optional[Any] = None,
        stopwords_path: Optional[str] = None,
        lsa_components: Optional[int] = None,
        synonyms_path: Optional[str] = None,
        semantic_embeddings_path: Optional[str] = None,
        semantic_threshold: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ):
        if system is not None:
            self.system = system
        else:
            from .system import PlagiarismDetectorSystem

            settings = get_settings()
            self.system = PlagiarismDetectorSystem(
                stopwords_path=stopwords_path or settings.stopwords_path,
                lsa_components=lsa_components or settings.lsa_components,
                synonyms_path=synonyms_path or settings.synonyms_path,
                semantic_embeddings_path=semantic_embeddings_path or settings.semantic_embeddings_path,
                semantic_threshold=(
                    semantic_threshold
                    if semantic_threshold is not None
                    else settings.semantic_threshold
                ),
                semantic_weight=(
                    semantic_weight
                    if semantic_weight is not None
                    else settings.semantic_weight
                ),
            )

    def __getattr__(self, name: str):
        return getattr(self.system, name)

    def read_document(self, filepath: str, preview_mode: bool = False) -> str:
        return self.system.read_document(filepath, preview_mode=preview_mode)

    def clean_academic_noise(self, text: str) -> str:
        return self.system.clean_academic_noise(text)

    def compare_files(
        self,
        target_file: str,
        reference_files: List[str],
        body_mode: bool = False,
    ):
        return self.system.check_similarity(
            target_file,
            reference_files,
            body_mode=body_mode,
        )
