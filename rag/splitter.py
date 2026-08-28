import re
from langchain_core.documents import Document

# Matches "Article 21.", "Article 21A.", "Article 368." etc. at line start
ARTICLE_PATTERN = re.compile(r"(?=^Article\s+(\d+[A-Z]?)\.?\s)", re.MULTILINE)

# Matches "PART III", "PART IVA" section headers used in the Constitution
PART_PATTERN = re.compile(r"PART\s+([IVXLC]+[A-Z]?)\s*[-–—]?\s*(.*)", re.IGNORECASE)


def _current_part(text_before: str) -> str:
    """Find the most recent PART header appearing before this chunk."""
    matches = list(PART_PATTERN.finditer(text_before))
    if not matches:
        return "Unknown"
    last = matches[-1]
    return f"Part {last.group(1)} - {last.group(2).strip()}" if last.group(2) else f"Part {last.group(1)}"


def split_constitution(docs: list[Document]) -> list[Document]:
    """
    Splits Constitution text by Article boundaries instead of raw character count.
    Falls back to a plain paragraph split for any leftover preamble/schedule text.
    """
    # Merge all pages into one string per source, keeping track of source name
    by_source: dict[str, str] = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        by_source[src] = by_source.get(src, "") + "\n" + d.page_content

    chunks: list[Document] = []

    for src, full_text in by_source.items():
        pieces = ARTICLE_PATTERN.split(full_text)
        # pieces alternates: [preamble_text, article_num, article_text, article_num, article_text, ...]
        preamble = pieces[0]
        if preamble.strip():
            chunks.append(Document(
                page_content=preamble.strip(),
                metadata={"source": src, "article_number": None, "part": "Preamble/Front matter"},
            ))

        for i in range(1, len(pieces), 2):
            article_num = pieces[i]
            article_text = pieces[i + 1] if i + 1 < len(pieces) else ""
            part = _current_part(full_text[: full_text.find(article_text)])

            chunks.append(Document(
                page_content=f"Article {article_num}. {article_text.strip()}",
                metadata={
                    "source": src,
                    "article_number": article_num,
                    "part": part,
                },
            ))

    # Assign chunk_id per source for citation display
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_id"] = counters[src]

    return chunks


# Keep the old generic splitter for non-Constitution docs (bare acts, notes, etc.)
def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_id"] = counters[src]
        c.metadata.setdefault("article_number", None)
        c.metadata.setdefault("part", None)
    return chunks