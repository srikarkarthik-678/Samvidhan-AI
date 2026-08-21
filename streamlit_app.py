import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("📚 AI Research Assistant — Agentic RAG")

with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDFs / text files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Index Documents") and uploaded:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded]
        with st.spinner("Indexing..."):
            resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.ok:
            st.success(f"Indexed {resp.json()['chunks_added']} chunks")
        else:
            st.error(resp.text)

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is retrieving, grading, and verifying..."):
            resp = requests.post(f"{API_URL}/ask", json={"question": question})
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            if data["citations"]:
                st.caption("Sources: " + ", ".join(data["citations"]))
            if not data["verified"]:
                st.warning("Answer could not be fully verified against source documents.")
            st.session_state.history.append(("assistant", data["answer"]))
        else:
            st.error(resp.text)