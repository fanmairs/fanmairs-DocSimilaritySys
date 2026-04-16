import re
from typing import Iterable, List


_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_LATIN_WORD_RE = re.compile(r'[A-Za-z][A-Za-z+\-]{1,}')
_NUMBER_RE = re.compile(r'(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?%?')
_DECIMAL_RE = re.compile(r'(?<![A-Za-z0-9_])-?\d+\.\d+%?')
_PERCENT_RE = re.compile(r'\d+(?:\.\d+)?%')
_NATURAL_SENTENCE_PUNCT_RE = re.compile(r'[。！？；;]|(?<!\d)[.!?](?!\d)')
_REPEATED_QUARTER_RE = re.compile(r'(?:\d{2}Q[1-4]\s*){3,}', re.IGNORECASE)
_REPEATED_YEAR_RE = re.compile(r'(?:20\d{2}\s*年?\s*){4,}')
_TABLE_KEYWORD_RE = re.compile(
    r'R\s*\^?\s*2|R²|MAE|RMSE|MSE|VIF|Beta|ARIMA|SARIMAX|BiLSTM|BiGRU|LSTM|GRU|CNN|TCN|Transformer|TF-CNN',
    re.IGNORECASE,
)
_STAT_TEST_RE = re.compile(r'\bp\s*[=<]|F\s*=|D-W|标准误|标准化系数|非标准化系数')
_NUMERIC_ONLY_BLOCK_RE = re.compile(r'[\d.,%Qq年\s|+\-*/=()<>]+')


def _compact_nonspace(text: str) -> str:
    return re.sub(r'\s+', '', text or '')


def is_numeric_table_noise(text: str) -> bool:
    """Return True for table/formula/chart fragments that should not be plagiarism evidence."""
    if not text or not text.strip():
        return False

    normalized = re.sub(r'\s+', ' ', text).strip()
    compact = _compact_nonspace(normalized)
    total = len(compact)
    if total == 0:
        return False

    cjk_count = len(_CJK_RE.findall(compact))
    digit_count = sum(ch.isdigit() for ch in compact)
    number_tokens = _NUMBER_RE.findall(normalized)
    decimal_tokens = _DECIMAL_RE.findall(normalized)
    percent_tokens = _PERCENT_RE.findall(normalized)
    latin_words = _LATIN_WORD_RE.findall(normalized)
    table_keywords = _TABLE_KEYWORD_RE.findall(normalized)
    natural_punct = _NATURAL_SENTENCE_PUNCT_RE.findall(normalized)
    stat_markers = _STAT_TEST_RE.findall(normalized)

    cjk_ratio = cjk_count / total
    digit_ratio = digit_count / total
    number_count = len(number_tokens)
    decimal_count = len(decimal_tokens)
    percent_count = len(percent_tokens)
    keyword_count = len(table_keywords) + len(stat_markers)

    # Keep regular English technical prose; table rows usually have few words and many numbers.
    has_english_sentence_shape = (
        cjk_count == 0
        and len(latin_words) >= 4
        and keyword_count < 2
    )
    if has_english_sentence_shape:
        return False

    if total <= 3 and digit_count > 0:
        return True

    if digit_count > 0 and cjk_count == 0 and _NUMERIC_ONLY_BLOCK_RE.fullmatch(normalized):
        return True

    if _REPEATED_QUARTER_RE.search(compact) or _REPEATED_YEAR_RE.search(compact):
        return True

    if keyword_count >= 3 and not natural_punct and cjk_ratio < 0.45:
        return True

    if percent_count >= 4 and number_count >= 4 and not natural_punct:
        return True

    if number_count >= 6 and not natural_punct:
        if decimal_count >= 3 or digit_ratio >= 0.25:
            return True

    if number_count >= 4 and decimal_count >= 3 and digit_ratio >= 0.20 and cjk_ratio < 0.40:
        return True

    math_symbol_count = sum(normalized.count(ch) for ch in ("=", "+", "-", "*", "/", "±", "≈", "<", ">"))
    if number_count >= 5 and math_symbol_count >= 3 and not natural_punct and cjk_ratio < 0.40:
        return True

    if cjk_count == 0 and number_count >= 3 and digit_ratio >= 0.42 and len(latin_words) <= 3:
        return True

    return False


def filter_detection_text_blocks(blocks: Iterable[str]) -> List[str]:
    """Drop non-narrative numeric/table blocks before detection text is assembled."""
    return [
        block
        for block in blocks
        if block and block.strip() and not is_numeric_table_noise(block)
    ]


# Compatibility hand-off: keep old imports working while the implementation
# now lives under text_processing.cleaners.
from text_processing.cleaners.noise import (  # noqa: E402,F401
    filter_detection_text_blocks,
    is_numeric_table_noise,
)
