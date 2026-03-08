"""
Helper utilities for the Medical RAG Assistant.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean document text: normalize whitespace and remove excessive newlines.
    """
    if not text or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def doc_id_from_path(file_path: Path, chunk_index: int) -> str:
    """
    Generate a unique ID for a document chunk for vector store upsert.
    """
    path_str = str(file_path.resolve())
    raw = f"{path_str}_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def extract_metadata_for_doc(doc: Document) -> dict[str, Any]:
    """
    Extract source and page from a Document for citation display.
    """
    meta = doc.metadata or {}
    source = meta.get("source", "Unknown")
    page = meta.get("page")
    if isinstance(source, (Path, str)):
        source = Path(source).name if isinstance(source, Path) else source
    return {"source": source, "page": page}
