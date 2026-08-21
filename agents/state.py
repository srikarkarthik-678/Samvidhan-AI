from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    retrieval_needed: bool
    grade_pass: bool
    rewrite_count: int
    verify_pass: bool
    verify_count: int
    verify_reason: str
    final_answer: str
    citations: List[str]