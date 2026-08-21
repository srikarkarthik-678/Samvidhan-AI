from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Add a per-source chunk index for clean citations
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_id"] = counters[src]

    return chunks