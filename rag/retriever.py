import re
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.splitter import split_constitution, split_documents
from rag.embeddings import get_embeddings
from rag.loader import load_from_directory
from rag.splitter import split_documents

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "research_docs"
ARTICLE_QUERY_PATTERN = re.compile(r"article\s+(\d+[A-Z]?)", re.IGNORECASE)

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def index_directory(directory: str | None = None) -> int:
    directory = directory or os.getenv("DOCS_DIR", "./data/documents")
    docs = load_from_directory(directory)
    if not docs:
        return 0

    chunks = split_documents(docs)
    if not chunks:
        return 0  # same guard here
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def index_documents(docs: list[Document]) -> int:
    chunks = split_documents(docs)
    if not chunks:
        return 0  # nothing extractable — don't call Chroma with empty embeddings
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)




def get_retriever(k: int = 5):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def index_constitution(docs) -> int:
    chunks = split_constitution(docs)
    if not chunks:
        return 0
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def hybrid_retrieve(question: str, k: int = 5):
    """If the question names a specific Article, filter exactly on it first;
    otherwise fall back to normal semantic search."""
    match = ARTICLE_QUERY_PATTERN.search(question)
    vs = get_vectorstore()

    if match:
        article_num = match.group(1)
        results = vs.get(where={"article_number": article_num})
        if results and results.get("documents"):
            from langchain_core.documents import Document
            return [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        # fall through to semantic search if exact match found nothing

    return vs.as_retriever(search_kwargs={"k": k}).invoke(question)