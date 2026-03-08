# Medical RAG Q&A Assistant

A production-style **Retrieval Augmented Generation (RAG)** application that lets users ask medical questions and get answers grounded in trusted medical documents. Built with **LangChain**, **OpenAI**, **Pinecone**, and **Streamlit**.

## Project Overview

The Medical Knowledge Assistant:

1. **Ingests** trusted medical documents (PDF, TXT) from a `data/` folder
2. **Splits** documents into optimized chunks (800 chars, 150 overlap)
3. **Embeds** chunks using OpenAI embeddings
4. **Stores** embeddings in a Pinecone vector database
5. **Retrieves** relevant chunks for each user question
6. **Generates** answers via OpenAI, using only retrieved context
7. **Shows** citations (document name + page number) and an interactive chat UI

The system is designed to **minimize hallucinations** by answering only from retrieved context and responding with *"I could not find information in the provided documents"* when context is insufficient.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                            │
│  Chat input │ Upload PDF │ Clear chat │ Re-index │ View indexed docs     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline (rag_pipeline.py)                   │
│  run_rag_stream(question) → retrieve → prompt → LLM stream → sources    │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ document_    │    │ retriever.py     │    │ prompt_template  │
│ loader.py    │    │ (Pinecone query) │    │ (strict prompt)  │
│ chunking.py  │    │ embeddings.py    │    │ ChatOpenAI       │
│ vector_store │    │                  │    │                  │
└──────────────┘    └─────────────────┘    └──────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pinecone (vector index)                               │
│  Namespace: medical-docs │ Metadata: source, page, file_name, text       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow:**

- **Indexing:** `data/sample_medical_docs` (and uploads) → load → clean → chunk → embed → Pinecone upsert
- **Query:** User question → embed query → Pinecone top-k → format context → LLM → streamed answer + sources

---

## Installation

### 1. Clone and enter project

```bash
cd Medical-Rag-QA
cd medical_rag_assistant
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment setup

Copy the example env file and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=medical-rag-index
PINECONE_NAMESPACE=medical-docs
```

Required:

- **OPENAI_API_KEY** – OpenAI API key (embeddings + chat)
- **PINECONE_API_KEY** – Pinecone API key
- **PINECONE_ENVIRONMENT** – e.g. `us-east-1`

Optional (have defaults):

- **PINECONE_INDEX_NAME** – default `medical-rag-index`
- **PINECONE_NAMESPACE** – default `medical-docs`
- **OPENAI_EMBEDDING_MODEL** – default `text-embedding-3-small`
- **OPENAI_CHAT_MODEL** – default `gpt-4o-mini`

---

## Running the App

From the `medical_rag_assistant` directory:

```bash
streamlit run app.py
```

Or from the repo root:

```bash
streamlit run medical_rag_assistant/app.py
```

The app will:

- Open in the browser (e.g. http://localhost:8501)
- On first question, load and index documents from `data/sample_medical_docs` (if the folder exists and contains PDF/TXT files)
- Let you ask questions, upload PDFs, re-index, and clear chat

---

## Adding Medical Documents

1. **Preloaded base:** Put PDF and/or TXT files in:
   ```
   medical_rag_assistant/data/sample_medical_docs/
   ```
   They are picked up when you run the app and trigger indexing (first query or “Re-index documents”).

2. **Upload at runtime:** Use the sidebar **“Upload new document”** to add a PDF. It is processed, chunked, embedded, and upserted into Pinecone so it is searchable immediately.

---

## Example Queries

- “What are the symptoms of dengue?”
- “Summarize the treatment guidelines for hypertension.”
- “Which medications are recommended in the document?”

Answers are based only on the indexed documents; citations show **document name** and **page number** (e.g. `WHO_Dengue_Guidelines.pdf – Page 12`).

---

## Project Structure

```
medical_rag_assistant/
├── app.py                 # Streamlit entry: chat, upload, sidebar
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── sample_medical_docs/   # PDF/TXT medical documents
├── rag/
│   ├── document_loader.py     # PDF/TXT loading
│   ├── chunking.py           # RecursiveCharacterTextSplitter
│   ├── embeddings.py         # OpenAI embeddings
│   ├── vector_store.py       # Pinecone index create/upsert
│   ├── retriever.py          # Semantic search
│   ├── prompt_template.py    # RAG prompt + format_context
│   └── rag_pipeline.py       # Index + run_rag + streaming
└── utils/
    ├── config.py             # Env and constants
    └── helpers.py            # clean_text, doc_id, metadata
```

---

## Features Summary

| Feature | Description |
|--------|-------------|
| Preloaded knowledge base | Auto load from `data/sample_medical_docs` (PDF/TXT) |
| User uploads | Upload PDFs in UI → process, chunk, embed, store in Pinecone |
| RAG pipeline | Load → clean → chunk → embed → store → retrieve → prompt → generate |
| Chunking | RecursiveCharacterTextSplitter (800 / 150) with source/page metadata |
| Embeddings | OpenAI (e.g. text-embedding-3-small) |
| Vector DB | Pinecone with auto index creation, namespace, metadata |
| Retrieval | Top-k (e.g. 5) semantic search |
| Hallucination mitigation | Answer only from context; “I could not find…” when insufficient |
| Citations | Document name and page number for each answer |
| Chat UI | Streamlit: chat history, streaming, upload, clear, re-index, view indexed docs |
| Disclaimer | Medical disclaimer at top of interface |

---

## Future Improvements

- **Evaluation logging:** Log queries, retrieved doc IDs, and model responses for evaluation and debugging
- **Vector index persistence:** Optional local vector store (e.g. FAISS) for development without Pinecone
- **Hybrid search:** Combine semantic and keyword (e.g. BM25) retrieval
- **Re-ranking:** Use a cross-encoder or small reranker on top-k before LLM
- **Conversation memory:** Optional multi-turn context (e.g. last N turns) for follow-up questions
- **Auth and rate limiting:** Simple API key or auth for deployment; rate limits per user/session

---

## License

Use and modify as needed for your project.
