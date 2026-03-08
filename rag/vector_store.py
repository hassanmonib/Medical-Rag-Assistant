"""
Pinecone vector store: index creation, upsert, and query.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

from utils.config import (
    EMBEDDING_DIMENSION,
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)
from utils.helpers import doc_id_from_path

# Cache index and client so we don't call list_indexes() on every query
_cached_index = None
_cached_client = None


def get_pinecone_client() -> Pinecone:
    """Return Pinecone client (cached)."""
    global _cached_client
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set")
    if _cached_client is None:
        _cached_client = Pinecone(api_key=PINECONE_API_KEY)
    return _cached_client


def ensure_index_exists() -> None:
    """
    Create Pinecone index if it does not exist.
    Uses serverless spec with cosine metric.
    """
    pc = get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_index():
    """Return Pinecone index (cached after first call to avoid list_indexes on every query)."""
    global _cached_index
    if _cached_index is None:
        ensure_index_exists()
        _cached_index = get_pinecone_client().Index(PINECONE_INDEX_NAME)
    return _cached_index


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embeddings for consistency with vector store."""
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def upsert_documents(
    documents: List[Document],
    embeddings: List[List[float]],
    namespace: str = PINECONE_NAMESPACE,
) -> int:
    """
    Upsert document chunks and their embeddings into Pinecone.
    Returns the number of vectors upserted.
    """
    if not documents or len(documents) != len(embeddings):
        return 0

    index = get_index()
    vectors = []
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        meta = doc.metadata.copy()
        # Store chunk text so we can reconstruct Document on retrieval
        meta["text"] = doc.page_content[:40_000]  # Pinecone metadata value size limit
        # Pinecone metadata values must be str, int, float, bool or list of these
        for k, v in list(meta.items()):
            if v is None:
                meta[k] = ""
            elif isinstance(v, (int, float, bool)):
                continue
            else:
                meta[k] = str(v)
        doc_id = doc_id_from_path(Path(meta.get("source", "") or ""), i)
        vectors.append({"id": doc_id, "values": emb, "metadata": meta})

    # Batch upsert (Pinecone accepts up to 100 per upsert)
    batch_size = 100
    for j in range(0, len(vectors), batch_size):
        batch = vectors[j : j + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    return len(vectors)
