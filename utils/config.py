"""
Configuration management for the Medical RAG Assistant.
Loads settings from environment variables with validation.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from app root and repo root
_app_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_app_root / ".env")
load_dotenv(dotenv_path=_app_root.parent / ".env")


# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR: Path = DATA_DIR / "sample_medical_docs"

# Chunking
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# Retrieval
TOP_K: int = 5

# Supported file extensions
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt"}

# Pinecone
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "medical-rag-index")
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "medical-docs")
PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")

# OpenAI
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Embedding dimensions for text-embedding-3-small
EMBEDDING_DIMENSION: int = 1536


def validate_config() -> list[str]:
    """
    Validate required environment variables.
    Returns a list of missing or invalid config keys.
    """
    errors: list[str] = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY")
    if not PINECONE_ENVIRONMENT:
        errors.append("PINECONE_ENVIRONMENT")
    return errors
