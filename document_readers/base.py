from typing import Protocol


class DocumentReader(Protocol):
    """Common callable shape for document readers."""

    def __call__(self, filepath: str, preview_mode: bool = False) -> str:
        ...


class UnsupportedDocumentTypeError(ValueError):
    """Raised when no reader is registered for a file extension."""
