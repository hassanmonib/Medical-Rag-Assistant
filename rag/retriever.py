"""
Semantic search retrieval using Pinecone.
Returns top-k relevant chunks with metadata for citations.
"""

from typing import List

from langchain_core.documents import Document

from rag.embeddings import embed_query
from rag.vector_store import get_index
from utils.config import PINECONE_NAMESPACE, TOP_K


def retrieve(query: str, top_k: int = TOP_K, namespace: str = PINECONE_NAMESPACE) -> List[Document]:
    """
    Retrieve top-k most relevant document chunks for the query.
    """
    if not query or not query.strip():
        return []

    index = get_index()
    query_embedding = embed_query(query)
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    docs = []
    for match in (response.matches or []):
        meta = (match.metadata or {}).copy()
        text = meta.pop("text", "") or ""
        if "source" not in meta and "file_name" in meta:
            meta["source"] = meta["file_name"]
        docs.append(Document(page_content=text or "[No content]", metadata=meta))
    return docs
