from .reader import _read_pdf_with_hybrid_layout


def read_pdf_with_hybrid(filepath: str) -> str:
    """Read PDF detection text using PyMuPDF blocks plus pdfplumber tables."""
    return _read_pdf_with_hybrid_layout(filepath)
