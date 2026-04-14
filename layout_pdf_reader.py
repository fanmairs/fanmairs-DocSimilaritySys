import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from grobid_client import read_pdf_body_with_grobid
from text_noise_filter import filter_detection_text_blocks, is_numeric_table_noise


BBox = Tuple[float, float, float, float]


@dataclass
class LayoutBlock:
    text: str
    page_index: int
    bbox: BBox
    page_width: float
    page_height: float
    kind: str = "body"


def _normalize_bbox(bbox: Iterable[float]) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _inflate_bbox(bbox: BBox, margin: float) -> BBox:
    x0, y0, x1, y1 = bbox
    return x0 - margin, y0 - margin, x1 + margin, y1 + margin


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return width * height


def _bbox_overlap_ratio(block_bbox: BBox, region_bbox: BBox) -> float:
    block_area = _bbox_area(block_bbox)
    if block_area <= 0:
        return 0.0
    return _bbox_intersection_area(block_bbox, region_bbox) / block_area


def _extract_pdfplumber_table_bboxes(filepath: str) -> Dict[int, List[BBox]]:
    try:
        import pdfplumber
    except ImportError:
        return {}

    table_bboxes: Dict[int, List[BBox]] = {}
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_bboxes = []
                for table in page.find_tables():
                    raw_bbox = getattr(table, "bbox", None)
                    if raw_bbox and len(raw_bbox) == 4:
                        page_bboxes.append(_inflate_bbox(_normalize_bbox(raw_bbox), 2.0))
                if page_bboxes:
                    table_bboxes[page_index] = page_bboxes
    except Exception:
        return {}
    return table_bboxes


def _extract_block_text(block: Dict) -> str:
    lines = []
    for line in block.get("lines", []):
        span_text = "".join(
            span.get("text", "")
            for span in line.get("spans", [])
            if span.get("text")
        ).strip()
        if span_text:
            lines.append(span_text)
    return "\n".join(lines).strip()


def _extract_pymupdf_layout_blocks(filepath: str) -> List[LayoutBlock]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("Please install PyMuPDF: pip install PyMuPDF") from exc

    blocks: List[LayoutBlock] = []
    with fitz.open(filepath) as doc:
        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict", sort=True)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                text = _extract_block_text(block)
                if not text:
                    continue
                raw_bbox = block.get("bbox")
                if not raw_bbox or len(raw_bbox) != 4:
                    continue
                blocks.append(
                    LayoutBlock(
                        text=text,
                        page_index=page_index,
                        bbox=_normalize_bbox(raw_bbox),
                        page_width=page_width,
                        page_height=page_height,
                    )
                )
    return blocks


def _is_page_number(text: str) -> bool:
    normalized = re.sub(r'\s+', '', text or "")
    return bool(re.fullmatch(r'-?\d{1,4}-?', normalized))


def _is_header_footer(block: LayoutBlock) -> bool:
    text = re.sub(r'\s+', ' ', block.text).strip()
    if not text:
        return True

    _, y0, _, y1 = block.bbox
    top_zone = block.page_height * 0.075
    bottom_zone = block.page_height * 0.925

    if _is_page_number(text) and (y0 < top_zone or y1 > bottom_zone):
        return True

    header_footer_terms = ("知网", "cnki", "学位论文", "硕士研究生学位论文")
    if (y0 < top_zone or y1 > bottom_zone) and any(term.lower() in text.lower() for term in header_footer_terms):
        return True

    return False


def _is_caption_or_source(text: str) -> bool:
    normalized = re.sub(r'\s+', ' ', text or "").strip()
    if not normalized:
        return False
    if re.match(r'^(图|表)\s*\d+(?:[-.]\d+)*\s+', normalized):
        return True
    if re.match(r'^(Figure|Fig\.|Table)\s*\d+(?:[-.]\d+)*\b', normalized, flags=re.IGNORECASE):
        return True
    if normalized.startswith("数据来源") or normalized.startswith("资料来源"):
        return True
    return False


def _is_formula_like(text: str) -> bool:
    normalized = re.sub(r'\s+', '', text or "")
    if len(normalized) < 8:
        return False

    math_symbols = sum(normalized.count(ch) for ch in ("=", "+", "-", "*", "/", "(", ")", "<", ">"))
    latin_or_digit = len(re.findall(r'[A-Za-z0-9]', normalized))
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', normalized))
    return math_symbols >= 3 and latin_or_digit >= 6 and cjk_count / max(1, len(normalized)) < 0.35


def _overlaps_any_table(block: LayoutBlock, table_bboxes: List[BBox]) -> bool:
    return any(_bbox_overlap_ratio(block.bbox, table_bbox) >= 0.55 for table_bbox in table_bboxes)


def classify_layout_blocks(
    blocks: List[LayoutBlock],
    table_bboxes_by_page: Optional[Dict[int, List[BBox]]] = None,
) -> List[LayoutBlock]:
    table_bboxes_by_page = table_bboxes_by_page or {}
    classified = []

    for block in blocks:
        table_bboxes = table_bboxes_by_page.get(block.page_index, [])
        if _is_header_footer(block):
            block.kind = "header_footer"
        elif table_bboxes and _overlaps_any_table(block, table_bboxes):
            block.kind = "table"
        elif _is_caption_or_source(block.text):
            block.kind = "caption"
        elif is_numeric_table_noise(block.text):
            block.kind = "numeric_noise"
        elif _is_formula_like(block.text):
            block.kind = "formula"
        else:
            block.kind = "body"
        classified.append(block)

    return classified


def _join_detection_blocks(blocks: List[LayoutBlock]) -> str:
    kept_texts = [
        block.text
        for block in blocks
        if block.kind == "body" and block.text.strip()
    ]
    # One last line-level pass catches chart labels split into tiny blocks.
    kept_texts = filter_detection_text_blocks(kept_texts)
    return "\n\n".join(kept_texts).strip()


def _read_pdf_basic_for_detection(filepath: str) -> str:
    blocks = _extract_pymupdf_layout_blocks(filepath)
    classified = classify_layout_blocks(blocks, table_bboxes_by_page={})
    return _join_detection_blocks(classified)


def _read_pdf_with_hybrid_layout(filepath: str) -> str:
    table_bboxes_by_page = _extract_pdfplumber_table_bboxes(filepath)
    blocks = _extract_pymupdf_layout_blocks(filepath)
    classified = classify_layout_blocks(blocks, table_bboxes_by_page=table_bboxes_by_page)
    text = _join_detection_blocks(classified)
    if text:
        return text
    return _read_pdf_basic_for_detection(filepath)


def _read_pdf_with_docling(filepath: str) -> str:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("docling is not installed") from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = os.getenv("DOCSIM_DOCLING_OCR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    pipeline_options.do_table_structure = os.getenv(
        "DOCSIM_DOCLING_TABLE_STRUCTURE",
        "0",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH")
    if artifacts_path:
        pipeline_options.artifacts_path = artifacts_path

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(filepath)
    document = getattr(result, "document", None)
    if document is None:
        return ""

    if hasattr(document, "export_to_markdown"):
        text = document.export_to_markdown()
    elif hasattr(document, "export_to_text"):
        text = document.export_to_text()
    else:
        text = str(document)

    lines = [line for line in text.splitlines() if line.strip()]
    lines = filter_detection_text_blocks(lines)
    return "\n\n".join(lines).strip()


def _read_pdf_with_grobid(filepath: str) -> str:
    return read_pdf_body_with_grobid(filepath)


def read_pdf_for_detection(filepath: str, backend: Optional[str] = None) -> str:
    """Read PDF body text for similarity detection with layout-aware filtering.

    Backends:
    - hybrid: PyMuPDF text blocks + pdfplumber table bounding boxes. Default.
    - pymupdf: PyMuPDF only, still with block classification.
    - docling: Docling conversion first, then fallback to hybrid.
    - grobid: GROBID TEI body extraction first, then fallback to hybrid.
    """
    selected_backend = (backend or os.getenv("DOCSIM_PDF_BACKEND") or "hybrid").strip().lower()

    if selected_backend == "grobid":
        try:
            text = _read_pdf_with_grobid(filepath)
            if text:
                return text
        except Exception:
            pass
        return _read_pdf_with_hybrid_layout(filepath)

    if selected_backend == "docling":
        try:
            text = _read_pdf_with_docling(filepath)
            if text:
                return text
        except Exception:
            pass
        return _read_pdf_with_hybrid_layout(filepath)

    if selected_backend == "pymupdf":
        return _read_pdf_basic_for_detection(filepath)

    return _read_pdf_with_hybrid_layout(filepath)
