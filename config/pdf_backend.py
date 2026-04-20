from __future__ import annotations

from typing import Optional


PDF_BACKEND_HYBRID = "hybrid"
PDF_BACKEND_PYMUPDF = "pymupdf"
PDF_BACKEND_DOCLING = "docling"
PDF_BACKEND_GROBID = "grobid"

DEFAULT_PDF_BACKEND = PDF_BACKEND_HYBRID
SUPPORTED_PDF_BACKENDS = {
    PDF_BACKEND_HYBRID,
    PDF_BACKEND_PYMUPDF,
    PDF_BACKEND_DOCLING,
    PDF_BACKEND_GROBID,
}

DEFAULT_GROBID_URL = "http://127.0.0.1:8070"
DEFAULT_GROBID_TIMEOUT = 45.0


def resolve_pdf_backend(value: Optional[str]) -> str:
    normalized = (value or DEFAULT_PDF_BACKEND).strip().lower()
    if normalized in SUPPORTED_PDF_BACKENDS:
        return normalized
    return DEFAULT_PDF_BACKEND
