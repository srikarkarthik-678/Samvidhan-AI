# AI Research Assistant — Agentic RAG

An AI research assistant that answers questions over your own PDFs/documents. Instead of a static "retrieve top-k, stuff into prompt" RAG pipeline, this uses a **LangGraph agent** that decides how to handle each question:

- Skips retrieval entirely for questions that don't need the document corpus
- Grades retrieved chunks for relevance instead of trusting top-k blindly
- Rewrites and retries the query if nothing relevant comes back
- Verifies the generated answer is actually grounded in the retrieved context before returning it

```
route → retrieve → grade docs → (rewrite + retry | generate) → verify → (regenerate | finalize)
```

## Tech Stack

- **LangChain** + **LangGraph** — orchestration and agent graph
- **ChromaDB** — vector store
- **OpenAI** (GPT + embeddings) — LLM and embedding provider (Gemini supported as a swap)
- **FastAPI** — backend API
- **Streamlit** — chat UI

## Project Structure

```
agentic-rag/
│
├── app.py                  # FastAPI backend (upload, reindex, ask, health)
├── streamlit_app.py        # Streamlit chat UI
│
├── agents/
│   ├── graph.py             # LangGraph graph definition
│   ├── nodes.py             # Node logic: route, retrieve, grade, rewrite, generate, verify
│   └── state.py              # Shared agent state schema
│
├── rag/
│   ├── loader.py             # Loads PDFs / txt / md into LangChain Documents
│   ├── splitter.py           # Chunks documents
│   ├── embeddings.py         # Embedding provider setup
│   └── retriever.py          # Vector store + retriever setup
│
├── data/
│   └── documents/            # Drop source documents here (or upload via UI)
│
├── chroma_db/                # Persisted vector store (auto-created)
├── .env                       # API keys and config
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- An OpenAI API key with billing enabled ([platform.openai.com/api-keys](https://platform.openai.com/api-keys)) — or a Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) if using Gemini instead

## Setup

### 1. Clone / create the project and enter it

```bash
cd agentic-rag
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script with an execution-policy error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Your terminal prompt should show `(venv)` once it's active.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create/edit `.env` in the project root:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
CHROMA_DIR=./chroma_db
DOCS_DIR=./data/documents
```

To use Gemini instead, set `LLM_PROVIDER=gemini`, `EMBEDDING_PROVIDER=gemini`, and fill in `GOOGLE_API_KEY` / `GEMINI_MODEL`.

## Adding Documents

Supported formats: `.pdf`, `.txt`, `.md` (scanned/image-only PDFs won't extract text).

**Option A — via the Streamlit UI:** upload files in the sidebar and click "Index Documents" (once the app is running — see below).

**Option B — manually:** drop files into `data/documents/`, then index from the command line:

```bash
python -c "from rag.retriever import index_directory; print(index_directory('./data/documents'))"
```

This prints the number of chunks indexed.

## Running the App

You need **two terminals**, both with the venv activated.

**Terminal 1 — backend:**
```bash
uvicorn app:app --reload --port 8000
```
Verify it's up at [http://localhost:8000/health](http://localhost:8000/health) → should return `{"status":"ok"}`. Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — frontend:**
```bash
streamlit run streamlit_app.py
```
Opens at [http://localhost:8501](http://localhost:8501). Upload documents and start asking questions.

## API Endpoints

| Method | Endpoint    | Description                                  |
|--------|-------------|-----------------------------------------------|
| GET    | `/health`   | Health check                                   |
| POST   | `/upload`   | Upload and index one or more documents         |
| POST   | `/reindex`  | Re-index everything currently in `DOCS_DIR`    |
| POST   | `/ask`      | Ask a question — body: `{"question": "..."}`    |

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: agents` / `rag` | Run `uvicorn` / `streamlit` from the project root, not from inside a subfolder |
| `streamlit`/`uvicorn` not recognized | Venv isn't activated in that terminal — activate it, then reinstall if needed |
| `venv\Scripts\Activate.ps1` not found | The venv wasn't created — run `python -m venv venv` first |
| `AuthenticationError` from OpenAI | Check `.env` has the correct key and that billing is enabled on your OpenAI account |
| Streamlit: "connection refused" | Backend isn't running — start `uvicorn` first, in a separate terminal |
| Answers say "no relevant context found" | Nothing has been indexed yet — add documents and index them (see above) |
| `ImportError: cannot import name 'Chroma'` | Run `pip install -U langchain-chroma` |

## Notes

- `chroma_db/` and `data/documents/` are created automatically and will hold your indexed data — exclude them from git if you don't want to commit indexed content.
- Each new terminal needs the venv reactivated — it doesn't persist across terminal sessions.
