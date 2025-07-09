import streamlit as st
import os
import uuid
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.pdf_extractor import extract_text_pdf
from backend.utils.chunk_embed import chunk_text, embed_store
from backend.utils.query import retrieve_chunks, answer_with_context_groq

DATA_DIR = "data"
Path(DATA_DIR).mkdir(exist_ok=True)

st.title("FusionFlow — Multi-Modal RAG")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    session_id = f"doc_{uuid.uuid4().hex[:8]}"
    st.session_state["session_id"] = session_id

    save_path = Path(DATA_DIR) / f"{session_id}_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    status_box = st.empty()
    status_box.info("Extracting text...")
    text = extract_text_pdf(open(save_path, "rb"))
    status_box.success("✅ Text extracted")

    status_box = st.empty()
    status_box.info("Chunking...")
    chunks = chunk_text(text)
    status_box.success("✅ Chunking done")

    status_box = st.empty()
    status_box.info("Indexing (embedding + storing)...")
    embed_store(chunks, session_id)
    status_box.success(f"✅ Embedded in session: {session_id}")

    st.subheader("Session ID")
    st.code(session_id)

    st.markdown("---")

    st.subheader("Ask a question based on the uploaded PDF")
    user_query = st.text_input("Your question")

    if user_query:
        with st.spinner("Retrieving and answering..."):
            top_chunks = retrieve_chunks(user_query, session_id)
            answer = answer_with_context_groq(user_query, top_chunks)
        st.success("Answer:")
        st.write(answer)
