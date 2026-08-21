import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

load_dotenv()

from agents.graph import compiled_graph
from rag.retriever import index_directory, index_documents
from rag.loader import load_documents

DOCS_DIR = os.getenv("DOCS_DIR", "./data/documents")
os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI(title="Agentic RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    verified: bool


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = os.path.join(DOCS_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    docs = load_documents(saved_paths)
    if not docs:
        raise HTTPException(400, "No supported documents found (.pdf, .txt, .md)")

    chunk_count = index_documents(docs)
    return {"indexed_files": [f.filename for f in files], "chunks_added": chunk_count}


@app.post("/reindex")
async def reindex():
    chunk_count = index_directory(DOCS_DIR)
    return {"chunks_indexed": chunk_count}


@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    result = compiled_graph.invoke({"question": req.question})
    return QueryResponse(
        answer=result["final_answer"],
        citations=result.get("citations", []),
        verified=result.get("verify_pass", True),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)