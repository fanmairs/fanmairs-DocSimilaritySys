import os
from typing import Callable, Dict

from .base import UnsupportedDocumentTypeError
from .docx.reader import read_docx_document
from .pdf.pymupdf_backend import read_pdf_preview
from .pdf.reader import read_pdf_for_detection
from .txt.reader import read_txt_document
from text_processing.normalizers.pdf import normalize_pdf_detection_text


Reader = Callable[[str, bool], str]


def _read_pdf_document(filepath: str, preview_mode: bool = False) -> str:
    if preview_mode:
        return read_pdf_preview(filepath)
    return normalize_pdf_detection_text(read_pdf_for_detection(filepath))


READERS: Dict[str, Reader] = {
    ".txt": read_txt_document,
    ".doc": read_docx_document,
    ".docx": read_docx_document,
    ".pdf": _read_pdf_document,
}


def read_document_by_type(filepath: str, preview_mode: bool = False) -> str:
    """Dispatch document reading by file extension."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    reader = READERS.get(ext)
    if reader is None:
        supported = ", ".join(sorted(READERS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported document format: {ext}. Supported formats: {supported}"
        )

    try:
        return reader(filepath, preview_mode=preview_mode).strip()
    except Exception as exc:
        raise ValueError(f"Failed to read document ({filepath}): {exc}") from exc
