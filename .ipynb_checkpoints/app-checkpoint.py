from fastapi import FastAPI
from pydantic import BaseModel

import chromadb
import ollama

from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# CONFIG MODEL
# ---------------------------------------------------

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI()

# ---------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------
# CHROMADB CLIENT
# ---------------------------------------------------

client = chromadb.Client()

# Create or get collection
collection = client.get_or_create_collection(name="rag_docs")

# ---------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------

class QueryRequest(BaseModel):
    question: str

# ---------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------

with open("documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------------------------------------------
# CHUNKING
# ---------------------------------------------------

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# ---------------------------------------------------
# STORE EMBEDDINGS ONLY IF EMPTY
# ---------------------------------------------------

if collection.count() == 0:

    for i, chunk in enumerate(chunks):

        # Convert chunk -> embedding vector
        embedding = embedding_model.encode(chunk).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

# ---------------------------------------------------
# HOME ROUTE
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "RAG Server Running",
        "total_chunks": len(chunks),
        "chunks": chunks
    }

# ---------------------------------------------------
# SEARCH + GENERATION ROUTE
# ---------------------------------------------------

@app.post("/search")
def search(request: QueryRequest):

    question = request.question

    # -----------------------------------------------
    # QUESTION -> EMBEDDING
    # -----------------------------------------------

    query_embedding = embedding_model.encode(question).tolist()

    # -----------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    retrieved_docs = results["documents"][0]

    # -----------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------

    context = "\n".join(retrieved_docs)

    # -----------------------------------------------
    # PROMPT
    # -----------------------------------------------

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

    # -----------------------------------------------
    # OLLAMA GENERATION
    # -----------------------------------------------

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # -----------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------

    return {
        "question": question,
        "retrieved_context": retrieved_docs,
        "answer": response["message"]["content"]
    }