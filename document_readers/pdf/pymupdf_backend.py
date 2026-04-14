from .reader import _read_pdf_basic_for_detection


def read_pdf_preview(filepath: str) -> str:
    """Read PDF text blocks for UI preview without detection-time filtering."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("Please install PyMuPDF: pip install PyMuPDF") from exc

    text_blocks = []
    with fitz.open(filepath) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            text_blocks.extend(block[4] for block in blocks if block[-1] == 0)
    return "\n".join(text_blocks).strip()


def read_pdf_with_pymupdf(filepath: str) -> str:
    """Read PDF detection text using only PyMuPDF layout blocks."""
    return _read_pdf_basic_for_detection(filepath)
