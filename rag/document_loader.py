"""
Document loading for PDF and TXT files.
Loads from a directory and supports single-file loading for uploads.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.config import SAMPLE_DOCS_DIR, SUPPORTED_EXTENSIONS


def load_document(file_path: Path) -> List[Document]:
    """
    Load a single document (PDF or TXT) and return list of Document objects.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use {SUPPORTED_EXTENSIONS}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = file_path.name
    return docs


def load_documents_from_directory(directory: Path | None = None) -> List[Document]:
    """
    Load all supported documents from a directory (default: sample_medical_docs).
    """
    directory = directory or SAMPLE_DOCS_DIR
    directory = Path(directory)
    if not directory.is_dir():
        return []

    all_docs: List[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(path)
                all_docs.extend(docs)
            except Exception as e:
                # Log but continue with other files
                print(f"Warning: Could not load {path}: {e}")
    return all_docs
