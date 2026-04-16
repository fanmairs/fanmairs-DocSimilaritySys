"""Text processing package for cleanup, normalization, and segmentation."""

from .cleaners.academic import clean_academic_noise
from .cleaners.noise import filter_detection_text_blocks, is_numeric_table_noise
from .normalizers.pdf import normalize_pdf_detection_text

__all__ = [
    "clean_academic_noise",
    "filter_detection_text_blocks",
    "is_numeric_table_noise",
    "normalize_pdf_detection_text",
]
