import re


_ACADEMIC_META_RE = re.compile(
    r"(学校代码|学号|分类号|中图分类号|基金项目|DOI|收稿日期|修回日期|录用日期|"
    r"作者简介|通讯作者)\s*[:：]?\s*[^\n。；;]*",
    flags=re.IGNORECASE,
)
_THESIS_LABEL_RE = re.compile(
    r"(硕士学位论文|博士学位论文|专业学位硕士学位论文|Dissertation|Thesis)",
    flags=re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,\-~]\s*\d+)*\s*\]")
_FIGURE_TABLE_TITLE_RE = re.compile(
    r"(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+[.\-]?\d*\s+[^\n。；;]{0,30}(?=[。；;\n]|$)",
    flags=re.IGNORECASE,
)
_FIGURE_TABLE_INLINE_RE = re.compile(
    r"[(（]?(?:如|见)?(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+[.\-]?\d*(?:所示)?[)）]?",
    flags=re.IGNORECASE,
)
_DISPLAY_FORMULA_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", flags=re.DOTALL)
_CHAPTER_NUMBER_RE = re.compile(r"(?:\d+\.){2,}\d+")
_TOC_LINE_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z]+[\s.·…]+[0-9IVX]{1,3}")
_NATURAL_SENTENCE_END_RE = re.compile(r"(?<=[。！？；;])")


def _cut_main_body(text: str) -> str:
    start_markers = [
        "引言",
        "绪论",
        "前言",
        "第一章",
        "1 引言",
        "1. 引言",
        "一、引言",
    ]
    end_markers = [
        "参考文献",
        "References",
        "致谢",
        "Acknowledgements",
        "附录",
        "Appendix",
        "攻读学位期间取得的成果",
        "攻读硕士学位期间发表的论文",
    ]

    start_positions = [text.find(marker) for marker in start_markers if text.find(marker) != -1]
    if not start_positions:
        return text

    start_idx = min(start_positions)
    end_positions = [
        text.find(marker, start_idx + 1)
        for marker in end_markers
        if text.find(marker, start_idx + 1) != -1
    ]
    end_idx = min(end_positions) if end_positions else len(text)
    if end_idx - start_idx > 1000:
        return text[start_idx:end_idx]
    return text


def _keep_natural_chinese_segments(text: str) -> str:
    segments = _NATURAL_SENTENCE_END_RE.split(text)
    kept = []
    for segment in segments:
        candidate = segment.strip()
        if not candidate:
            continue
        chinese_count = len(re.findall(r"[\u4e00-\u9fa5]", candidate))
        latin_count = len(re.findall(r"[A-Za-z]", candidate))
        if chinese_count >= 5 and chinese_count >= latin_count * 0.5:
            kept.append(candidate)
    return "".join(kept) if kept else text


def clean_academic_noise(text: str) -> str:
    """Clean common academic front matter, citations, figures, formulas, and TOC residue."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""

    cleaned = _ACADEMIC_META_RE.sub(" ", cleaned)
    cleaned = _THESIS_LABEL_RE.sub(" ", cleaned)
    cleaned = _cut_main_body(cleaned)
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _FIGURE_TABLE_TITLE_RE.sub(" ", cleaned)
    cleaned = _FIGURE_TABLE_INLINE_RE.sub("", cleaned)
    cleaned = _DISPLAY_FORMULA_RE.sub(" ", cleaned)
    cleaned = _TOC_LINE_RE.sub(" ", cleaned)
    cleaned = _CHAPTER_NUMBER_RE.sub(" ", cleaned)
    cleaned = _keep_natural_chinese_segments(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
