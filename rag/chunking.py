"""
Chunking strategy using LangChain RecursiveCharacterTextSplitter.
Preserves metadata: source, page, file_name.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.helpers import clean_text


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with overlap.
    Preserves source, page, and file_name in metadata.
    """
    if not documents:
        return []

    for doc in documents:
        if doc.page_content:
            doc.page_content = clean_text(doc.page_content)

    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    return chunks
