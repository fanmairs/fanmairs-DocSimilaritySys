def read_docx_document(filepath: str, preview_mode: bool = False) -> str:
    """Read Word text. Legacy .doc files still require conversion upstream."""
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("Please install python-docx: pip install python-docx") from exc

    doc = Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs).strip()
