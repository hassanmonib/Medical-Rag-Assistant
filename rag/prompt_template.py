"""
Strict prompt template for RAG to minimize hallucination.
Instructs the LLM to answer only from context and cite sources.
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM = """You are a medical knowledge assistant. You answer questions using ONLY the provided context from trusted medical documents.

Rules:
- Answer ONLY using the context below. Do not use external knowledge.
- If the context does not contain enough information to answer the question, respond exactly: "I could not find information in the provided documents to answer this question."
- When you use information from the context, cite the source as: [Document name – Page X].
- Be concise and accurate. Do not add information that is not in the context.
- If the question is not related to the provided context, say you can only answer based on the loaded documents."""

RAG_USER = """Context:
{context}

Question:
{question}

Answer using only the provided context. Include source citations (document name and page number) where applicable."""


def get_rag_prompt() -> ChatPromptTemplate:
    """Return the RAG chat prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM),
            ("human", RAG_USER),
        ]
    )


def format_context(docs: list) -> str:
    """
    Format retrieved documents as a single context string with source labels.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("file_name") or meta.get("source", "Unknown")
        page = meta.get("page", "")
        page_str = f" (Page {page})" if page not in (None, "") else ""
        parts.append(f"[{source}{page_str}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
