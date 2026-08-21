import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import get_embeddings
from rag.loader import load_from_directory
from rag.splitter import split_documents

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "research_docs"


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def index_directory(directory: str | None = None) -> int:
    """Load, chunk, and index all documents in a directory. Returns chunk count."""
    directory = directory or os.getenv("DOCS_DIR", "./data/documents")
    docs = load_from_directory(directory)
    if not docs:
        return 0

    chunks = split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def index_documents(docs: list[Document]) -> int:
    chunks = split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int = 5):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})