"""Streamlit chat UI for the RAG Document Q&A backend."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # seconds; embedding a big PDF can be slow

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def _error_message(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def api_upload(file) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/upload",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(_error_message(response))
    return response.json()


def api_ask(question: str, doc_id: str | None) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question, "doc_id": doc_id},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(_error_message(response))
    return response.json()


def api_health() -> dict | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])  # [{"role", "content", "sources"}]
st.session_state.setdefault("documents", {})  # {doc_id: filename}
st.session_state.setdefault("active_doc", None)  # doc_id or None (= all)
st.session_state.setdefault("processed_files", set())  # avoid re-uploading on rerun


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**{i}. {src['filename']} — page {src['page_number']}** "
                f"· similarity {src['score']:.2f}"
            )
            st.caption(src["snippet"])


# --------------------------------------------------------------------------- #
# Sidebar: upload + document picker
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📄 Documents")

    health = api_health()
    if health:
        st.success(f"API online · {health['indexed_chunks']} chunks indexed")
    else:
        st.error(f"API unreachable at {API_BASE_URL}")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None:
        file_key = f"{uploaded.name}:{uploaded.size}"
        if file_key not in st.session_state.processed_files:
            with st.spinner("Reading, chunking and embedding…"):
                try:
                    result = api_upload(uploaded)
                except (RuntimeError, requests.RequestException) as exc:
                    st.error(f"Upload failed: {exc}")
                else:
                    st.session_state.processed_files.add(file_key)
                    st.session_state.documents[result["doc_id"]] = result["filename"]
                    st.session_state.active_doc = result["doc_id"]
                    st.success(
                        f"Indexed **{result['filename']}** "
                        f"({result['num_pages']} pages, {result['num_chunks']} chunks)"
                    )

    if st.session_state.documents:
        options = {"All documents": None, **{n: d for d, n in st.session_state.documents.items()}}
        labels = list(options.keys())
        current = next(
            (n for n, d in options.items() if d == st.session_state.active_doc), labels[0]
        )
        choice = st.selectbox("Ask about", labels, index=labels.index(current))
        st.session_state.active_doc = options[choice]

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main: chat
# --------------------------------------------------------------------------- #
st.title("Chat with your PDF")
st.caption("Answers come only from your document, with page citations.")

if not st.session_state.documents:
    st.info("👈 Upload a PDF in the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your document…", disabled=not health)
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = api_ask(question, st.session_state.active_doc)
                answer, sources = reply["answer"], reply["sources"]
            except (RuntimeError, requests.RequestException) as exc:
                answer, sources = f"⚠️ {exc}", []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
