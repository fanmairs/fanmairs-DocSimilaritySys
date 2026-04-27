import re
from typing import Iterable, List, Optional, Tuple


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SENTENCE_END_RE = re.compile(r"(?<=[。！？；;])")
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,\-~]\s*\d+)*\s*\]")
_DISPLAY_FORMULA_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", flags=re.DOTALL)
_CHAPTER_NUMBER_RE = re.compile(r"\b(?:\d+\.){2,}\d+\b")
_DOT_LEADER_RE = re.compile(r"(?:\.{3,}|…{2,}|·{3,})")
_TOC_LINE_RE = re.compile(r"^\s*.+(?:\.{3,}|…{2,}|·{3,}).{0,30}\d{1,4}\s*$")
_STANDALONE_PAGE_RE = re.compile(r"^\s*(?:[IVXLCDM]{1,8}|\d{1,4})\s*$", flags=re.IGNORECASE)

_META_LINE_RE = re.compile(
    r"(学校代码|学号|分类号|中图分类号|密级|UDC|基金项目|DOI|收稿日期|修回日期|录用日期|"
    r"作者简介|通讯作者|作者姓名|导师姓名|指导教师|校外指导教师|学科.*专业|专业.*领域|"
    r"论文答辩|答辩委员会|学生所属学院|院系|班级)"
)
_THESIS_LABEL_RE = re.compile(
    r"(硕士学位论文|博士学位论文|专业学位硕士学位论文|本科毕业论文|毕业设计|"
    r"Dissertation|Thesis)",
    flags=re.IGNORECASE,
)
_FIGURE_TABLE_TITLE_RE = re.compile(
    r"^\s*(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+(?:[-.]\d+)*\s+.{0,90}$",
    flags=re.IGNORECASE,
)
_FIGURE_TABLE_INLINE_RE = re.compile(
    r"[（(]?(?:如|见|详见)?(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+(?:[-.]\d+)*"
    r"(?:所示)?[)）]?",
    flags=re.IGNORECASE,
)

_BODY_START_RE = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百千万\d]+\s*章\s*(?:绪论|引言|前言|概述)?|"
    r"1\s*[.、 ]\s*(?:绪论|引言|前言|研究背景)|"
    r"(?:绪论|引言|前言))$"
)
_BODY_END_RE = re.compile(
    r"^(?:参考文献|致谢|谢辞|附录|攻读学位期间取得的成果|攻读硕士学位期间发表的论文|"
    r"References|Acknowledgements?|Appendix)$",
    flags=re.IGNORECASE,
)


def _normalize_linebreaks(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_output_spacing(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text or "")
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _line_offsets(text: str) -> Iterable[Tuple[int, str]]:
    position = 0
    for line in text.splitlines(keepends=True):
        yield position, line
        position += len(line)


def _compact_heading(line: str) -> str:
    return re.sub(r"\s+", "", (line or "").strip())


def _is_toc_like_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if _TOC_LINE_RE.match(stripped):
        return True
    return bool(_DOT_LEADER_RE.search(stripped) and re.search(r"\d{1,4}\s*$", stripped))


def _is_metadata_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if len(stripped) <= 140 and _META_LINE_RE.search(stripped):
        return True
    return len(stripped) <= 80 and bool(_THESIS_LABEL_RE.search(stripped))


def _remove_metadata_lines(text: str) -> str:
    kept: List[str] = []
    previous_removed = False
    for line in text.splitlines():
        stripped = line.strip()
        if _is_metadata_line(stripped):
            previous_removed = True
            continue
        if previous_removed and len(stripped) <= 80 and re.match(r"^[:：]", stripped):
            continue
        kept.append(line)
        previous_removed = False
    return "\n".join(kept)


def _body_start_score(text: str, position: int) -> float:
    window = text[position : position + 1800]
    cjk_count = len(_CJK_RE.findall(window))
    sentence_count = len(re.findall(r"[。！？；]", window))
    toc_noise = len(_DOT_LEADER_RE.findall(window))
    return min(cjk_count / 80.0, 15.0) + sentence_count * 2.5 - toc_noise * 8.0


def _find_body_start(text: str) -> Optional[int]:
    candidates: List[Tuple[float, int]] = []
    for position, line in _line_offsets(text):
        stripped = line.strip()
        if not stripped or _is_toc_like_line(stripped):
            continue
        compact = _compact_heading(stripped)
        if len(compact) > 32:
            continue
        if _BODY_START_RE.match(compact):
            score = _body_start_score(text, position)
            if score > 3.0:
                candidates.append((score, position))

    if not candidates:
        return None
    # Prefer the first plausible body heading; TOC headings have already been rejected.
    return min(position for _, position in candidates)


def _is_body_end_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped or _is_toc_like_line(stripped):
        return False
    compact = _compact_heading(stripped)
    return len(compact) <= 32 and bool(_BODY_END_RE.match(compact))


def _find_body_end(text: str, start: int) -> Optional[int]:
    for position, line in _line_offsets(text):
        if position <= start + 1000:
            continue
        if _is_body_end_line(line):
            return position
    return None


def _cut_main_body(text: str) -> str:
    start = _find_body_start(text)
    start_index = start if start is not None else 0
    end = _find_body_end(text, start_index)
    end_index = end if end is not None else len(text)
    if end_index - start_index > 1000:
        return text[start_index:end_index].strip()
    return text.strip()


def _remove_structural_noise_lines(text: str) -> str:
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _is_toc_like_line(stripped):
            continue
        if _STANDALONE_PAGE_RE.match(stripped):
            continue
        if _FIGURE_TABLE_TITLE_RE.match(stripped) and len(stripped) <= 120:
            continue
        kept.append(line)
    return "\n".join(kept)


def _keep_narrative_segments(text: str) -> str:
    paragraphs = re.split(r"\n{2,}", text or "")
    kept: List[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        cjk_count = len(_CJK_RE.findall(normalized))
        latin_count = len(re.findall(r"[A-Za-z]", normalized))
        digit_count = sum(ch.isdigit() for ch in normalized)
        if cjk_count >= 8 and cjk_count >= latin_count * 0.25:
            kept.append(normalized)
            continue
        if cjk_count >= 5 and digit_count <= max(6, cjk_count):
            kept.append(normalized)

    narrative = "\n\n".join(kept).strip()
    if len(narrative) >= max(300, len(text or "") * 0.25):
        return narrative
    return text


def _light_cleanup(text: str) -> str:
    cleaned = _remove_metadata_lines(_normalize_linebreaks(text))
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _DISPLAY_FORMULA_RE.sub(" ", cleaned)
    cleaned = _FIGURE_TABLE_INLINE_RE.sub("", cleaned)
    cleaned = _remove_structural_noise_lines(cleaned)
    cleaned = _CHAPTER_NUMBER_RE.sub(" ", cleaned)
    return _normalize_output_spacing(cleaned)


def _looks_overcleaned(raw_text: str, cleaned: str) -> bool:
    raw_len = len(re.sub(r"\s+", "", raw_text or ""))
    cleaned_len = len(re.sub(r"\s+", "", cleaned or ""))
    if raw_len < 3000:
        return False
    return cleaned_len < max(1000, raw_len * 0.15)


def clean_academic_noise(text: str) -> str:
    """Remove thesis front matter, TOC residue, references, formulas and captions.

    The cleaner keeps PDF line boundaries until after body extraction. This avoids
    treating a TOC entry such as "第一章 绪论 ... 1" as the real body start.
    """
    normalized = _normalize_linebreaks(text)
    if not normalized:
        return ""

    cleaned = _remove_metadata_lines(normalized)
    cleaned = _cut_main_body(cleaned)
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _DISPLAY_FORMULA_RE.sub(" ", cleaned)
    cleaned = _FIGURE_TABLE_INLINE_RE.sub("", cleaned)
    cleaned = _remove_structural_noise_lines(cleaned)
    cleaned = _CHAPTER_NUMBER_RE.sub(" ", cleaned)
    cleaned = _keep_narrative_segments(cleaned)
    cleaned = _normalize_output_spacing(cleaned)

    if _looks_overcleaned(normalized, cleaned):
        return _light_cleanup(normalized)
    return cleaned
