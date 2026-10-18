"""Streamlit UI for Mini RAG Document Q&A."""

import numpy as np
import streamlit as st

import rag

st.set_page_config(page_title="Mini RAG Document Q&A", page_icon="📄")


@st.cache_data(show_spinner=False)
def index_file(filename: str, data: bytes, embed_model: str):
    """Extract, chunk and embed one file. Cached, so re-asking questions is instant."""
    pages = rag.extract_pages(filename, data)
    chunks = rag.build_chunks(filename, pages)
    embeddings = rag.embed_texts([c["text"] for c in chunks], embed_model)
    return chunks, embeddings, len(pages)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Setup")
    st.markdown(
        "1. Install and start [Ollama](https://ollama.com)\n"
        f"2. `ollama pull {rag.CHAT_MODEL}`\n"
        f"3. `ollama pull {rag.EMBED_MODEL}`"
    )
    st.subheader("Models")
    installed = rag.installed_models()
    for label, name in (("Chat", rag.CHAT_MODEL), ("Embeddings", rag.EMBED_MODEL)):
        if installed is None:
            st.write(f"{label}: `{name}`")
        elif rag.model_installed(name, installed):
            st.write(f"✅ {label}: `{name}`")
        else:
            st.write(f"⚠️ {label}: `{name}` (not installed)")
    if installed is None:
        st.warning("Ollama is not reachable right now.")

    st.subheader("How it works")
    st.markdown(
        "1. Extract text from your files\n"
        "2. Split it into overlapping chunks\n"
        "3. Embed chunks locally\n"
        "4. Find the chunks most similar to your question (cosine similarity)\n"
        "5. Ask the local model to answer using only those chunks"
    )
    st.caption(
        "Everything runs on your machine. Grounded prompting reduces unsupported "
        "answers but cannot eliminate them, so check the sources."
    )

# ---------------- Main page ----------------
st.title("📄 Mini RAG Document Q&A")
st.caption("Ask questions about your PDF and TXT files. Answers come from your documents, with sources.")

uploaded = st.file_uploader(
    "Upload documents", type=["pdf", "txt"], accept_multiple_files=True
)
st.caption(f"Maximum {rag.MAX_UPLOAD_MB} MB per file. Scanned PDFs are not supported (no OCR).")

all_chunks, all_vectors, indexed_docs = [], [], []
for f in uploaded or []:
    data = f.getvalue()
    if len(data) > rag.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"{f.name}: file is larger than {rag.MAX_UPLOAD_MB} MB.")
        continue
    try:
        with st.spinner(f"Indexing {f.name}..."):
            chunks, vectors, n_pages = index_file(f.name, data, rag.EMBED_MODEL)
    except rag.RagError as exc:
        st.error(f"{f.name}: {exc}")
        continue
    all_chunks.extend(chunks)
    all_vectors.append(vectors)
    indexed_docs.append((f.name, n_pages, len(chunks)))

if not indexed_docs:
    st.info(
        "👆 Upload a PDF or TXT file to get started. "
        "You can try `sample_docs/sample.txt` from this project."
    )
    st.stop()

st.subheader("Indexed documents")
for name, n_pages, n_chunks in indexed_docs:
    pages = f"{n_pages} page(s), " if name.lower().endswith(".pdf") else ""
    st.write(f"📄 **{name}**: {pages}{n_chunks} chunks")

st.subheader("Ask a question")
question = st.text_input("Your question", placeholder="e.g. How long can I borrow a book?")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question first.")
        st.stop()
    try:
        with st.spinner("Searching your documents and generating an answer..."):
            matrix = np.vstack(all_vectors)
            hits = rag.retrieve(question, all_chunks, matrix)
            if hits[0][1] < rag.MIN_SCORE:
                # Heuristic guard: nothing looks relevant, so skip generation.
                answer = rag.NOT_FOUND
            else:
                answer = rag.answer_question(question, hits)
    except rag.RagError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    with st.expander("Retrieved evidence and sources"):
        st.caption(
            "Similarity is a simple heuristic (cosine similarity of embeddings), "
            "not a calibrated confidence score."
        )
        for n, (chunk, score) in enumerate(hits, start=1):
            page = f", page {chunk['page']}" if chunk["page"] else ""
            st.markdown(f"**[{n}] {chunk['source']}{page}** (similarity {score:.2f})")
            st.text(chunk["text"])  # st.text shows content as plain text, never as markdown/HTML
