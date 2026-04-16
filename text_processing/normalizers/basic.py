import re


def normalize_text(text: str) -> str:
    """Collapse all whitespace into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_paragraphs(text: str) -> str:
    """Normalize line endings while preserving paragraph separators."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
