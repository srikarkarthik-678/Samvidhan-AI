# Samvidhaan AI — Agentic RAG for the Constitution of India

An AI research assistant that helps law students look up and understand provisions of the Constitution of India. Instead of a static "retrieve top-k, stuff into prompt" RAG pipeline, this uses a **LangGraph agent** that decides how to handle each question:

- Skips retrieval entirely for questions that don't need the document corpus
- Uses **hybrid retrieval**: exact Article-number lookup when a question names a specific Article, falling back to semantic search for conceptual questions
- Grades retrieved chunks for relevance instead of trusting top-k blindly
- Rewrites and retries the query if nothing relevant comes back
- Verifies the generated answer is grounded in the retrieved context before returning it
- Preserves the Constitution's structure (Article number, Part) via structure-aware chunking, and cites answers according


## Tech Stack

- **LangChain** + **LangGraph** — orchestration and agent graph
- **ChromaDB** — vector store
- **OpenAI** (GPT + embeddings) — LLM and embedding provider (Gemini supported as a swap)
- **FastAPI** — backend API
- **Streamlit** — chat UI


## Prerequisites

- Python 3.10+
- An OpenAI API key with billing enabled ([platform.openai.com/api-keys](https://platform.openai.com/api-keys)) — or a Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) if using Gemini instead

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/srikarkarthik-678/Ai-Research.git
cd Ai-Research
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> If PowerShell blocks the activation script with an execution-policy error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
CHROMA_DIR=./chroma_db
DOCS_DIR=./data/documents
```

## Adding the Constitution

1. Get a clean, text-based (not scanned) PDF of the Constitution of India — e.g. from `legislative.gov.in` or `india.gov.in`
2. Save it as `data/documents/constitution_of_india.pdf` (filename must contain "constitution" to trigger Article-aware chunking)
3. Index it:

```bash
python -c "from rag.loader import load_documents; from rag.retriever import index_constitution; docs = load_documents(['./data/documents/constitution_of_india.pdf']); print(index_constitution(docs))"
```

This should print a number in the hundreds (the Constitution has ~470 Articles).

Other supported document types (`.pdf`, `.txt`, `.md`) get indexed with the generic splitter via `/upload` or `/reindex`.

## Running the App

You need **two terminals**, both with the venv activated.

**Terminal 1 — backend:**
```bash
uvicorn app:app --reload --port 8000
```
Verify at [http://localhost:8000/health](http://localhost:8000/health) → `{"status":"ok"}`. API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — frontend:**
```bash
streamlit run streamlit_app.py
```
Opens at [http://localhost:8501](http://localhost:8501).

## API Endpoints

| Method | Endpoint    | Description                                              |
|--------|-------------|------------------------------------------------------------|
| GET    | `/health`   | Health check                                                |
| POST   | `/upload`   | Upload and index documents (routes Constitution PDFs correctly) |
| POST   | `/reindex`  | Re-index everything currently in `DOCS_DIR`                 |
| POST   | `/ask`      | Ask a question — body: `{"question": "..."}`                |

## Example Queries

- `"What does Article 21 say?"` — exact-match retrieval by Article number
- `"Which article protects freedom of speech?"` — semantic/conceptual retrieval

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: agents` / `rag` | Run `uvicorn` / `streamlit` from the project root |
| `NameError: ARTICLE_QUERY_PATTERN not defined` | Ensure the regex constant is defined at module level in `retriever.py`, above any functions that use it |
| "No relevant context" for a known Article | File likely wasn't indexed via `index_constitution()` — check filename contains "constitution" and re-index; clear `chroma_db/` if old generic-chunked data exists |
| `AuthenticationError` from OpenAI | Check `.env` has the correct key and billing is enabled |
| Streamlit: "connection refused" | Start the FastAPI backend first |
| `ValueError: Expected Embeddings to be non-empty` | Uploaded file had no extractable text (likely a scanned/image-only PDF) |

## Disclaimer

This tool is for educational purposes only and does not constitute legal advice.
