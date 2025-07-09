import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./backend/rag/chroma_store")

def retrieve_chunks(query, session_id, top_k=4):
    collection = chroma_client.get_or_create_collection(name=session_id)
    query_embedding = model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Use the following context to answer the question.
    Context:{context}
    Question:{question}
    Answer the question with relevant to the given context
    """
)

def answer_with_context_groq(question, context_chunks):
    context = "\n\n".join(context_chunks)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
