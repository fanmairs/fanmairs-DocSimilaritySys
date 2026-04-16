import re
from typing import List

from text_processing.normalizers.basic import normalize_for_paragraphs, normalize_text


def get_paragraphs(text: str, min_chars: int = 50, max_count: int = 24) -> List[str]:
    """Extract paragraphs, falling back to long sentence windows if separators are missing."""
    normalized = normalize_for_paragraphs(text)
    if not normalized:
        return []

    paragraph_candidates = normalized.split("\n\n")
    if len(paragraph_candidates) > 1:
        paragraphs = [
            paragraph.strip()
            for paragraph in paragraph_candidates
            if len(paragraph.strip()) >= min_chars
        ]
        if paragraphs:
            return paragraphs[:max_count]

    flattened = normalize_text(text)
    if not flattened:
        return []

    fallback = []
    chunks = [
        sentence.strip()
        for sentence in re.findall(r"[^。！？；]+[。！？；]?", flattened)
        if sentence.strip()
    ]
    current = []
    current_len = 0
    for sentence in chunks:
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= min_chars:
            fallback.append("".join(current))
            current = []
            current_len = 0
        if len(fallback) >= max_count:
            break

    if current and len("".join(current)) >= min_chars and len(fallback) < max_count:
        fallback.append("".join(current))
    return fallback
