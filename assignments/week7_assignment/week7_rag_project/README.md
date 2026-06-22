# Local RAG Document Question Answering System

This project implements a local Retrieval-Augmented Generation (RAG) system that answers questions based on custom uploaded PDF documents. 

Unlike public APIs, this pipeline uses local models for parsing, chunking, embedding, and vector search, keeping your documents confidential. It uses the **Groq API** only for final answer generation (using open-source LLMs like Llama 3).

**Week 7 - Celebal Technologies Internship**  
**Author**: Khushi Bhagat

## Overview

A Retrieval-Augmented Generation (RAG) system bypasses the context limits and training cutoffs of LLMs by retrieving relevant sections of documents and feeding them as context. This application runs:
1. **Local Text Extraction**: Reads uploaded PDFs using PyMuPDF.
2. **Local Text Chunking**: Splits text into overlapping segments, preserving sentences.
3. **Local Vector Embeddings**: Generates 384-dimensional dense vectors using a lightweight Sentence-Transformers model (`all-MiniLM-L6-v2`) locally on your CPU/GPU.
4. **Local Vector Search**: Stores and indexes chunks inside an embedded vector database (**ChromaDB**) on your machine.
5. **Fast Generation**: Queries the **Groq API** with retrieved context blocks to compile a fact-grounded answer using `llama3-8b-8192`.

---

## Project Structure

```
├── src/
│   ├── app.py              # Streamlit frontend (entry point)
│   ├── vectorstore.py      # PDF loader, text chunking, ChromaDB integration
│   └── chatbot.py          # Groq LLM API integration
├── week7_khushi_bhagat.ipynb # Step-by-step Jupyter Notebook walkthrough
├── requirements.txt        # Python dependency specifications
├── .env.example            # Environment variables placeholder
├── .gitignore              # Git ignore rules
└── README.md               # This document
```

---

## Getting Started

### 1. Set Up Environment
Ensure you have Python 3.8+ installed. Clone or navigate to this folder and run:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and enter your Groq API Key:
```env
GROQ_API_KEY=your_actual_groq_api_key
```
You can generate a free key from the [Groq Console](https://console.groq.com/keys).

### 3. Run Streamlit App
Navigate to the project and boot the interface:
```bash
cd src
streamlit run app.py
```

---

## Jupyter Notebook Walkthrough
A complete, step-by-step notebook showing how each individual stage of the RAG pipeline is written, executed, and analyzed is available at:
`[week7_khushi_bhagat.ipynb](../week7_khushi_bhagat.ipynb)`

The cells in this notebook are fully decoupled:
- **Cell 1**: Installation and imports
- **Cell 2**: PyMuPDF text loader function
- **Cell 3**: Sentence-boundary chunking logic
- **Cell 4**: Local Sentence-Transformers initialization & encoding
- **Cell 5**: ChromaDB local database initialization and index creation
- **Cell 6**: Similarity-search query function
- **Cell 7**: Groq API chat compilation and prompt formatting
- **Cell 8**: End-to-end execution on a sample PDF

---

## System Workflow & Architecture

```
PDF File → Load Pages (PyMuPDF) → Segment Text (Chunking) → Local Embeddings (SentenceTransformers) → Local Storage (ChromaDB)
                                                                                                            ↓
User Query → Local Embeddings (SentenceTransformers) → Vector Similarity Search (ChromaDB) → Context Chunks retrieved
                                                                                                            ↓
Context Chunks + User Query → Prompt Formulation → LLM Completion (Groq Llama 3) → Fact-Grounded Answer
```

---

## Key Learnings & Student Summary
- **Retrieval-Augmented Generation**: Combines the precision of search retrieval with the fluency of modern language models.
- **Embedded Databases**: Local stores like ChromaDB make it easy to prototype systems with no backend database hosting bills or configuration delays.
- **Dense Vector Representations**: Vector similarity (e.g., Cosine Distance) accurately matches queries with documents based on meaning, rather than simple keyword matches.
- **Modularity in AI Pipelines**: Separating loading, chunking, embedding, storage, search, and generation steps allows independent testing and optimization of each component.
