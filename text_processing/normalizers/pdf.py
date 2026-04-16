import re


def normalize_pdf_detection_text(text: str) -> str:
    """Remove forced PDF line breaks while preserving paragraph boundaries."""
    normalized = re.sub(r"\n\s*\n", "<PARA_BREAK>", text or "")
    normalized = re.sub(r"\n", "", normalized)
    normalized = re.sub(r"<PARA_BREAK>", "\n\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()
