import os
import re
import uuid
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Optional

from text_processing.cleaners.noise import filter_detection_text_blocks


DEFAULT_GROBID_URL = "http://127.0.0.1:8070"


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _iter_text_blocks(element: ET.Element) -> List[str]:
    blocks = []
    skip_tags = {
        "table",
        "figure",
        "formula",
        "listBibl",
        "note",
    }

    def walk(node: ET.Element) -> None:
        tag = _strip_namespace(node.tag)
        if tag in skip_tags:
            return

        if tag in {"head", "p"}:
            text = " ".join("".join(node.itertext()).split())
            if text:
                blocks.append(text)
            return

        for child in list(node):
            walk(child)

    walk(element)
    return blocks


def extract_body_text_from_tei(tei_xml: str) -> str:
    if not tei_xml or not tei_xml.strip():
        return ""

    root = ET.fromstring(tei_xml)
    body = None
    for element in root.iter():
        if _strip_namespace(element.tag) == "body":
            body = element
            break

    if body is None:
        return ""

    blocks = _iter_text_blocks(body)
    blocks = filter_detection_text_blocks(blocks)
    return "\n\n".join(blocks).strip()


def _build_multipart_body(filepath: str, fields: Optional[dict] = None) -> tuple[bytes, str]:
    boundary = f"----DocSimilaritySys{uuid.uuid4().hex}"
    body_chunks: List[bytes] = []
    fields = fields or {}

    for name, value in fields.items():
        body_chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        body_chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body_chunks.append(str(value).encode("utf-8"))
        body_chunks.append(b"\r\n")

    filename = os.path.basename(filepath)
    body_chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    body_chunks.append(
        (
            'Content-Disposition: form-data; name="input"; '
            f'filename="{filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8")
    )
    with open(filepath, "rb") as pdf_file:
        body_chunks.append(pdf_file.read())
    body_chunks.append(b"\r\n")
    body_chunks.append(f"--{boundary}--\r\n".encode("utf-8"))

    return b"".join(body_chunks), boundary


def process_fulltext_document(
    filepath: str,
    grobid_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    base_url = (grobid_url or os.getenv("GROBID_URL") or DEFAULT_GROBID_URL).rstrip("/")
    endpoint = f"{base_url}/api/processFulltextDocument"
    timeout_seconds = float(timeout or os.getenv("GROBID_TIMEOUT", "45"))
    body, boundary = _build_multipart_body(
        filepath,
        fields={
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            "includeRawCitations": "0",
            "includeRawAffiliations": "0",
            "segmentSentences": "0",
        },
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/xml,text/xml",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"GROBID request failed: {exc}") from exc


def read_pdf_body_with_grobid(
    filepath: str,
    grobid_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    tei_xml = process_fulltext_document(filepath, grobid_url=grobid_url, timeout=timeout)
    text = extract_body_text_from_tei(tei_xml)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# Compatibility hand-off: keep old imports working while the implementation
# now lives under document_readers.pdf.
from document_readers.pdf.grobid_backend import (  # noqa: E402,F401
    DEFAULT_GROBID_URL,
    extract_body_text_from_tei,
    process_fulltext_document,
    read_pdf_body_with_grobid,
)
