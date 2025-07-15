# FusionFlow

**FusionFlow** is a multi-modal Retrieval-Augmented Generation (RAG) system that enables users to upload PDFs, extract and process their content of any form (including text, tables, and images), and ask questions based on the uploaded documents.

## Features

- **PDF Upload & Processing:** Extracts text, tables, and images from PDFs.
- **Multi-modal Extraction:** Uses OCR and image captioning for images within PDFs.
- **Chunking & Embedding:** Splits extracted content into chunks and generates vector embeddings.
- **Vector Database:** Stores embeddings in a persistent ChromaDB instance for efficient retrieval.
- **Contextual Q&A:** Retrieves relevant chunks and uses a language model (Groq Llama3) to answer user queries based on document context.
- **Streamlit Frontend:** User-friendly web interface for uploading documents and interacting with the system.

## Directory Structure

```
.
├── backend/
│   ├── main.py
│   ├── rag/
│   └── utils/
│       ├── pdf_extractor.py
│       ├── chunk_embed.py
│       ├── query.py
│       └── image_process.py
├── frontend/
│   ├── app.py
│   └── data/
├── data/
├── flow/
├── requirements.txt
├── LICENSE
├── README.md
└── Notes.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harikumar4/FusionFlow.git
   cd FusionFlow
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Additional dependencies (not in requirements.txt) may be required:
   - `torch`
   - `transformers`
   - `pytesseract`
   - `Pillow`
   - `python-dotenv`
   - `langchain_groq`
   - `pandas`
   - `fitz` (PyMuPDF)

   Install them as needed:
   ```bash
   pip install torch transformers pytesseract Pillow python-dotenv langchain_groq pandas pymupdf
   ```

   **Note:** Install Tesseract before running

3. **Set up environment variables:**
   - Create a `.env` file in the root directory for any required API keys (e.g., Groq).

## Usage

1. **Start the Streamlit frontend:**
   ```bash
   streamlit run frontend/app.py
   ```

2. **Upload a PDF:** Use the web interface to upload a PDF document.

3. **Ask Questions:** Once processed, enter your questions in the interface to get context-aware answers.

## How It Works?

1. **Extraction:** PDF content (text, tables, images) is extracted using PyMuPDF and pdfplumber.
2. **Image Processing:** Images are captioned using BLIP and OCR is performed with Tesseract.
3. **Chunking:** Text is split into manageable chunks.
4. **Embedding:** Chunks are embedded using Sentence Transformers.
5. **Storage:** Embeddings are stored in ChromaDB.
6. **Retrieval & Q&A:** On user query, relevant chunks are retrieved and passed to a language model for answer generation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.