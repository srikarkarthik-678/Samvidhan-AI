# Agentic RAG Research Assistant

LangGraph-based agent that decides how to retrieve, grade, rewrite, and verify
answers instead of running a static RAG pipeline.

## Setup

pip install -r requirements.txt
cp .env.example .env   # fill in API keys

## Run backend

uvicorn app:app --reload --port 8000

## Run frontend (separate terminal)

streamlit run streamlit_app.py

## Index documents via CLI (optional)

python -c "from rag.retriever import index_directory; index_directory('./data/documents')"

## Architecture

route → retrieve → grade → (rewrite+retry | generate) → verify → (regenerate | finalize)
