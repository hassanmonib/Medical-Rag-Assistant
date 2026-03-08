"""
End-to-end RAG pipeline: indexing, retrieve context, build prompt, generate answer.
Supports streaming and returns answer with sources for citations.
"""

from pathlib import Path
from typing import Generator, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.chunking import split_documents
from rag.document_loader import load_document, load_documents_from_directory
from rag.embeddings import embed_documents
from rag.prompt_template import format_context, get_rag_prompt
from rag.retriever import retrieve
from rag.vector_store import upsert_documents
from utils.config import OPENAI_CHAT_MODEL, SAMPLE_DOCS_DIR


def index_documents(documents: List[Document]) -> int:
    """
    Chunk, embed, and upsert documents into Pinecone.
    Returns number of chunks indexed.
    """
    if not documents:
        return 0
    chunks = split_documents(documents)
    if not chunks:
        return 0
    embeddings = embed_documents(chunks)
    return upsert_documents(chunks, embeddings)


def index_data_folder(directory: Optional[Path] = None) -> int:
    """
    Load all documents from data folder, chunk, embed, and upsert.
    Returns total chunks indexed.
    """
    directory = directory or SAMPLE_DOCS_DIR
    docs = load_documents_from_directory(directory)
    return index_documents(docs)


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """Return configured OpenAI chat model."""
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        streaming=streaming,
    )


def run_rag(
    question: str,
    top_k: int = 5,
    stream: bool = True,
) -> tuple[str, list[dict]]:
    """
    Run RAG: retrieve chunks, build prompt, generate answer.
    Returns (answer_text, list of source dicts with 'source' and 'page').
    """
    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        return (
            "I could not find information in the provided documents to answer this question.",
            [],
        )

    context = format_context(retrieved)
    prompt = get_rag_prompt()
    llm = get_llm(streaming=stream)

    messages = prompt.invoke({"context": context, "question": question})
    if stream:
        full_answer: list[str] = []
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_answer.append(chunk.content)
        answer_text = "".join(full_answer)
    else:
        response = llm.invoke(messages)
        answer_text = getattr(response, "content", "") or str(response)

    # Build unique sources from retrieved docs
    seen: set[tuple[str, Optional[int]]] = set()
    sources: list[dict] = []
    for doc in retrieved:
        meta = doc.metadata or {}
        name = meta.get("file_name") or meta.get("source") or "Unknown"
        page = meta.get("page")
        key = (str(name), page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": name, "page": page})

    return answer_text.strip(), sources


def run_rag_stream(
    question: str,
    top_k: int = 5,
) -> Generator[tuple[Optional[str], Optional[list[dict]]], None, None]:
    """
    Stream RAG response token by token. Yields (content_chunk, None) for each token,
    then (None, sources) at the end so the UI can show streaming text and then citations.
    """
    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        yield "I could not find information in the provided documents to answer this question.", None
        seen: set[tuple[str, Optional[int]]] = set()
        yield None, []
        return

    context = format_context(retrieved)
    prompt = get_rag_prompt()
    llm = get_llm(streaming=True)
    messages = prompt.invoke({"context": context, "question": question})

    seen = set()
    sources: list[dict] = []
    for doc in retrieved:
        meta = doc.metadata or {}
        name = meta.get("file_name") or meta.get("source") or "Unknown"
        page = meta.get("page")
        key = (str(name), page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": name, "page": page})

    for chunk in llm.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content, None
    yield None, sources
