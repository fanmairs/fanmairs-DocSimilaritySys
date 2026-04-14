from .reader import _read_pdf_with_docling


def read_pdf_with_docling(filepath: str) -> str:
    """Read PDF detection text through Docling."""
    return _read_pdf_with_docling(filepath)
