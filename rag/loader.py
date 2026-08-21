
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(paths: list[str]) -> list[Document]:
    """Load PDFs/text files into LangChain Document objects with source metadata."""
    docs: list[Document] = []

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {path}")
            continue

        if ext == ".pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")

        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)

    return docs


def load_from_directory(directory: str) -> list[Document]:
    paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return load_documents(paths)