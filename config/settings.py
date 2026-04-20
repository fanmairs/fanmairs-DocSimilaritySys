from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from . import paths
from .pdf_backend import (
    DEFAULT_GROBID_TIMEOUT,
    DEFAULT_GROBID_URL,
    DEFAULT_PDF_BACKEND,
    resolve_pdf_backend,
)


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in TRUE_VALUES


def _env_path(name: str, default: Path) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return str(default)
    return str(paths.resolve_project_path(value.strip()))


def _optional_env_path(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return str(paths.resolve_project_path(value.strip()))
    return None


@dataclass(frozen=True)
class AppSettings:
    host: str
    port: int
    temp_upload_dir: str
    task_db_file: str
    stopwords_path: str
    synonyms_path: str
    semantic_embeddings_path: str
    lsa_components: int
    semantic_threshold: float
    semantic_weight: float
    pdf_backend: str
    docling_ocr: bool
    docling_table_structure: bool
    docling_artifacts_path: Optional[str]
    grobid_url: str
    grobid_timeout: float

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            host=_env_str("DOCSIM_HOST", "0.0.0.0"),
            port=_env_int("DOCSIM_PORT", 8000),
            temp_upload_dir=_env_path("DOCSIM_TEMP_UPLOAD_DIR", paths.TEMP_UPLOAD_DIR),
            task_db_file=_env_path("DOCSIM_TASK_DB_FILE", paths.TASK_DB_FILE),
            stopwords_path=_env_path("DOCSIM_STOPWORDS_PATH", paths.STOPWORDS_PATH),
            synonyms_path=_env_path("DOCSIM_SYNONYMS_PATH", paths.SYNONYMS_PATH),
            semantic_embeddings_path=_env_path(
                "DOCSIM_SEMANTIC_EMBEDDINGS_PATH",
                paths.SEMANTIC_EMBEDDINGS_PATH,
            ),
            lsa_components=_env_int("DOCSIM_LSA_COMPONENTS", 3),
            semantic_threshold=_env_float("DOCSIM_SEMANTIC_THRESHOLD", 0.55),
            semantic_weight=_env_float("DOCSIM_SEMANTIC_WEIGHT", 0.35),
            pdf_backend=resolve_pdf_backend(_env_str("DOCSIM_PDF_BACKEND", DEFAULT_PDF_BACKEND)),
            docling_ocr=_env_bool("DOCSIM_DOCLING_OCR", False),
            docling_table_structure=_env_bool("DOCSIM_DOCLING_TABLE_STRUCTURE", False),
            docling_artifacts_path=_optional_env_path(
                "DOCSIM_DOCLING_ARTIFACTS_PATH",
                "DOCLING_ARTIFACTS_PATH",
            ),
            grobid_url=_env_str("GROBID_URL", DEFAULT_GROBID_URL),
            grobid_timeout=_env_float("GROBID_TIMEOUT", DEFAULT_GROBID_TIMEOUT),
        )


def get_settings() -> AppSettings:
    return AppSettings.from_env()
