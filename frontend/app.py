import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.pdf_extractor import extract_text_pdf, extract_images_pdf, extract_tables_pdf

st.title("Multi-Modal RAG")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_pdf(uploaded_file)
    st.subheader("Extracted Text")
    st.text_area("Text Output", text, height=400)

    uploaded_file.seek(0)
    images = extract_images_pdf(uploaded_file)

    if images:
        st.subheader("Extracted Images")
        for img in images:
            st.image(img)
    else:
        st.info("No images found in the PDF.")

    uploaded_file.seek(0)
    tables = extract_tables_pdf(uploaded_file)

    if tables:
        st.subheader("Extracted Tables")
        for i, table in enumerate(tables):
            st.write(f"Table {i+1}")
            st.dataframe(table)
    else:
        st.info("No tables found in the PDF.")
