from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    route_question, route_edge, direct_answer,
    retrieve, grade_documents, grade_edge, rewrite_query,
    generate, verify_answer, verify_edge, finalize,
)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_question", route_question)
    g.add_node("direct_answer", direct_answer)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("verify_answer", verify_answer)
    g.add_node("finalize", finalize)

    g.set_entry_point("route_question")

    g.add_conditional_edges("route_question", route_edge, {
        "retrieve": "retrieve",
        "direct_answer": "direct_answer",
    })

    g.add_edge("direct_answer", END)
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges("grade_documents", grade_edge, {
        "generate": "generate",
        "rewrite": "rewrite_query",
    })

    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "verify_answer")

    g.add_conditional_edges("verify_answer", verify_edge, {
        "generate": "generate",
        "finalize": "finalize",
    })

    g.add_edge("finalize", END)

    return g.compile()


# Compiled once, reused across requests
compiled_graph = build_graph()