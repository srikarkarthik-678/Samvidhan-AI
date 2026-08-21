import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from agents.state import AgentState
from rag.retriever import get_retriever

load_dotenv()

MAX_REWRITES = 2
MAX_VERIFY_RETRIES = 2


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"), temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


llm = get_llm()
retriever = get_retriever(k=5)


# ---------- 1. ROUTER ----------
route_prompt = ChatPromptTemplate.from_messages([
    ("system", "You decide if a user question requires searching an internal document corpus, "
               "or if it's general/conversational and can be answered directly. "
               "Respond with strict JSON only: {{\"needs_retrieval\": bool, \"reasoning\": str}}"),
    ("human", "{question}"),
])
router_chain = route_prompt | llm | JsonOutputParser()


def route_question(state: AgentState) -> AgentState:
    result = router_chain.invoke({"question": state["question"]})
    return {
        **state,
        "retrieval_needed": result["needs_retrieval"],
        "original_question": state["question"],
        "rewrite_count": 0,
        "verify_count": 0,
    }


def route_edge(state: AgentState) -> str:
    return "retrieve" if state["retrieval_needed"] else "direct_answer"


# ---------- 2. DIRECT ANSWER ----------
def direct_answer(state: AgentState) -> AgentState:
    resp = llm.invoke(state["question"])
    return {**state, "final_answer": resp.content, "citations": []}


# ---------- 3. RETRIEVE ----------
def retrieve(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


# ---------- 4. GRADE DOCUMENTS ----------
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade whether a retrieved document chunk is relevant to the question. "
               "Respond with strict JSON only: {{\"relevant\": true}} or {{\"relevant\": false}}."),
    ("human", "Question: {question}\n\nDocument chunk:\n{doc}"),
])
grade_chain = grade_prompt | llm | JsonOutputParser()


def grade_documents(state: AgentState) -> AgentState:
    relevant_docs = []
    for doc in state["documents"]:
        result = grade_chain.invoke({"question": state["question"], "doc": doc.page_content})
        if result.get("relevant"):
            relevant_docs.append(doc)

    return {**state, "documents": relevant_docs, "grade_pass": len(relevant_docs) > 0}


def grade_edge(state: AgentState) -> str:
    if state["grade_pass"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "generate"


# ---------- 5. REWRITE QUERY ----------
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question to be more effective for vector retrieval. "
               "Fix ambiguity, add likely synonyms, keep it concise. Return ONLY the rewritten question."),
    ("human", "Original question: {question}"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rewrite_query(state: AgentState) -> AgentState:
    new_q = rewrite_chain.invoke({"question": state["question"]})
    return {**state, "question": new_q.strip(), "rewrite_count": state["rewrite_count"] + 1}


# ---------- 6. GENERATE ----------
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question using ONLY the provided context chunks. "
               "Cite sources inline like [source:chunk_id]. "
               "If context is insufficient, say so explicitly rather than guessing."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])
generate_chain = generate_prompt | llm | StrOutputParser()


def generate(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"[{d.metadata.get('source')}:{d.metadata.get('chunk_id')}] {d.page_content}"
        for d in state["documents"]
    ) or "No relevant context found in the document corpus."

    answer = generate_chain.invoke({"question": state["original_question"], "context": context})
    citations = list({f"{d.metadata.get('source')}:{d.metadata.get('chunk_id')}" for d in state["documents"]})
    return {**state, "generation": answer, "citations": citations}


# ---------- 7. VERIFY ----------
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact-checker. Determine if the ANSWER is fully supported by the CONTEXT. "
               "Respond with strict JSON only: {{\"grounded\": bool, \"reason\": str}}"),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])
verify_chain = verify_prompt | llm | JsonOutputParser()


def verify_answer(state: AgentState) -> AgentState:
    context = "\n\n".join(d.page_content for d in state["documents"])
    result = verify_chain.invoke({"context": context, "answer": state["generation"]})
    return {
        **state,
        "verify_pass": result.get("grounded", False),
        "verify_reason": result.get("reason", ""),
        "verify_count": state["verify_count"] + 1,
    }


def verify_edge(state: AgentState) -> str:
    if state["verify_pass"] or state["verify_count"] >= MAX_VERIFY_RETRIES:
        return "finalize"
    return "generate"


# ---------- 8. FINALIZE ----------
def finalize(state: AgentState) -> AgentState:
    answer = state["generation"]
    if not state["verify_pass"]:
        answer += "\n\n⚠️ Note: parts of this answer could not be fully verified against the source documents."
    return {**state, "final_answer": answer}