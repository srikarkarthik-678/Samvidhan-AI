import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")