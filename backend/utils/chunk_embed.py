from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./backend/rag/chroma_store")

def chunk_text(content_blocks, chunk_size=500, chunk_overlap=50):
    if not content_blocks:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    for block in content_blocks:
        if block["type"] == "text":
            text_chunks = splitter.split_text(block["content"])
            chunks.extend([{"type": "text", "content": chunk, "page": block["page"]} for chunk in text_chunks])
        else:
            # For non-text content (images), keep them as is
            chunks.append(block)
    return chunks

def embed_store(chunks, collection_name):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Convert chunks to plain text for embedding while preserving metadata
    texts = [chunk["content"] for chunk in chunks]
    metadata = [{"type": chunk["type"], "page": chunk.get("page", -1)} for chunk in chunks]
    
    embeddings = model.encode(texts)
    ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        ids=ids
    )
