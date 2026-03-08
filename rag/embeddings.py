"""
OpenAI embeddings for document chunks and queries.
"""

from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from utils.config import OPENAI_EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    """Return configured OpenAI embeddings model."""
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
    )


def embed_documents(documents: List[Document]) -> List[List[float]]:
    """
    Generate embeddings for a list of documents.
    """
    if not documents:
        return []
    model = get_embedding_model()
    texts = [doc.page_content for doc in documents]
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    model = get_embedding_model()
    return model.embed_query(query)
