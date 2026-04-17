from typing import Any, List, Optional


class TraditionalEngine:
    """Adapter around the existing white-box plagiarism system."""

    kind = "traditional"

    def __init__(
        self,
        system: Optional[Any] = None,
        stopwords_path: str = "dicts/stopwords.txt",
        lsa_components: int = 3,
        synonyms_path: Optional[str] = "dicts/synonyms.txt",
        semantic_embeddings_path: Optional[str] = "dicts/embeddings/fasttext_zh.vec",
        semantic_threshold: float = 0.55,
        semantic_weight: float = 0.35,
    ):
        if system is not None:
            self.system = system
        else:
            from .system import PlagiarismDetectorSystem

            self.system = PlagiarismDetectorSystem(
                stopwords_path=stopwords_path,
                lsa_components=lsa_components,
                synonyms_path=synonyms_path,
                semantic_embeddings_path=semantic_embeddings_path,
                semantic_threshold=semantic_threshold,
                semantic_weight=semantic_weight,
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
