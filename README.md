# 🚀 Local RAG AI Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application built using:

- FastAPI
- ChromaDB
- Ollama
- Sentence Transformers

This project demonstrates the core architecture behind modern AI assistants and document-aware chat systems.

---

# 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines:

Semantic Search + Large Language Models

Instead of relying only on pretrained knowledge, the LLM retrieves relevant information from your documents before generating an answer.

---

# 🔥 Features

- FastAPI backend API
- Semantic search using embeddings
- Vector database integration with ChromaDB
- Local LLM inference using Ollama
- Prompt augmentation
- Environment-based model configuration
- Beginner-friendly architecture
- Fully local execution (no cloud required)

---

# 🏗️ Architecture

documents.txt
      ↓
Chunking
      ↓
Embeddings
      ↓
ChromaDB Vector Storage
      ↓
Semantic Retrieval
      ↓
Prompt Augmentation
      ↓
Ollama LLM
      ↓
Generated Response

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM Runtime | Ollama |
| LLM Model | Llama3 |
| Environment Config | python-dotenv |

---

# 📁 Project Structure

project/
│
├── app.py
├── documents.txt
├── .env
├── requirements.txt
└── README.md

---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd <project-folder>
