import re
from typing import Dict, List, Optional


def make_span(text: str, start: int, end: Optional[int] = None) -> Optional[Dict]:
    raw = text or ""
    if end is None:
        end = start + len(raw)

    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw) - len(raw.rstrip())
    span_start = start + left_trim
    span_end = end - right_trim
    if span_end <= span_start:
        return None

    cleaned = raw.strip()
    if not cleaned:
        return None

    return {"text": cleaned, "start": span_start, "end": span_end}


def split_sentences(text: str, min_chars: int = 1) -> List[str]:
    """Split text into natural sentence-like chunks."""
    if not text:
        return []

    raw_parts = re.split(r"([。？！；\n]+)", text)
    sentences = []
    for index in range(0, len(raw_parts) - 1, 2):
        sentence = (raw_parts[index] + raw_parts[index + 1]).strip()
        if len(sentence) >= min_chars:
            sentences.append(sentence)

    if len(raw_parts) % 2 != 0:
        tail = raw_parts[-1].strip()
        if len(tail) >= min_chars:
            sentences.append(tail)

    return sentences


def split_sentences_with_offsets(text: str) -> List[Dict[str, int]]:
    """Split text into sentence-like spans with original offsets."""
    if not text:
        return []

    pattern = re.compile(r"[^\u3002\uff01\uff1f\uff1b!?;\n]+[\u3002\uff01\uff1f\uff1b!?;\n]*")
    sentences = []
    for match in pattern.finditer(text):
        span = make_span(match.group(0), match.start(), match.end())
        if span is not None:
            sentences.append(span)

    if sentences:
        return sentences

    fallback = make_span(text, 0, len(text))
    return [fallback] if fallback else []
