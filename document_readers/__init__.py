"""Document reading package.

The public entry point is ``read_document_by_type``. Individual file formats
live in their own subpackages so PDF, DOCX, TXT, and future readers can evolve
independently.
"""

from .factory import read_document_by_type

__all__ = ["read_document_by_type"]
