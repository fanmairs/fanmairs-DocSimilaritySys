import math
import re
from typing import Dict, List, Optional, Tuple


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return float(1.0 / (1.0 + exp_term))

    exp_term = math.exp(value)
    return float(exp_term / (1.0 + exp_term))


def normalize_text(text: str) -> str:
    import re

    return re.sub(r'\s+', ' ', text or '').strip()


def normalize_for_paragraphs(text: str) -> str:
    import re

    normalized = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'[ \t\f\v]+', ' ', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


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


def split_sentences_with_offsets(text: str) -> List[Dict[str, int]]:
    if not text:
        return []

    pattern = re.compile(r'[^\u3002\uff01\uff1f\uff1b!?;\n]+[\u3002\uff01\uff1f\uff1b!?;\n]*')
    sentences = []
    for match in pattern.finditer(text):
        span = make_span(match.group(0), match.start(), match.end())
        if span is not None:
            sentences.append(span)

    if sentences:
        return sentences

    fallback = make_span(text, 0, len(text))
    return [fallback] if fallback else []


def get_paragraphs(text: str, min_chars: int = 50, max_count: int = 24) -> List[str]:
    normalized = normalize_for_paragraphs(text)
    if not normalized:
        return []

    paras = [p.strip() for p in normalized.split('\n\n') if len(p.strip()) >= min_chars]
    if paras:
        return paras[:max_count]

    # Fallback: when paragraph separators are missing, use long sentence windows.
    flattened = normalize_text(text)
    if not flattened:
        return []

    fallback = []
    pattern = r'[^。！？；]+[。！？；]?'
    import re

    chunks = [s.strip() for s in re.findall(pattern, flattened) if s.strip()]
    current = []
    current_len = 0
    for sent in chunks:
        current.append(sent)
        current_len += len(sent)
        if current_len >= min_chars:
            fallback.append(''.join(current))
            current = []
            current_len = 0
        if len(fallback) >= max_count:
            break

    if current and len(''.join(current)) >= min_chars and len(fallback) < max_count:
        fallback.append(''.join(current))
    return fallback


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def sum_intervals(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    return int(sum(max(0, end - start) for start, end in intervals))


def extract_entities(text: str):
    import re

    return set(re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', text))


def extract_tags(text: str):
    import jieba.analyse

    return set(jieba.analyse.extract_tags(text, topK=5))


def get_skeleton(text: str) -> str:
    import jieba.posseg as pseg

    words = pseg.cut(text)
    skeleton = [w.word for w in words if w.flag in ['v', 'p', 'c', 'd']]
    return "".join(skeleton)


def is_formula_explanation(text: str) -> bool:
    import re

    explanation_keywords = ['公式', '其中', '表示', '定义', '计算', '如图', '如式', '等于', '获得', '所示']
    keyword_count = sum(1 for kw in explanation_keywords if kw in text)
    has_math_symbols = len(re.findall(r'[A-Za-z]+|\d+', text)) > 5
    return keyword_count >= 2 and has_math_symbols


def safe_iou(set_a: set, set_b: set, default: float = 1.0) -> float:
    if not set_a and not set_b:
        return default
    union = len(set_a.union(set_b))
    if union == 0:
        return default
    return len(set_a.intersection(set_b)) / union
