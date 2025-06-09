# 📚 RAG_Pipeline (CLI + UI Based Document QA)

A local, memory-aware Retrieval-Augmented Generation (RAG) pipeline that:
- Parses and chunks `.pdf`, `.txt`, `.csv` documents
- Embeds them into a FAISS vector store
- Supports querying via CLI or Gradio UI
- Uses local LLM inference via [Ollama](https://ollama.com)
- Stores and retrieves chat memory for continuity

---

## Features

- Semantic retrieval from your uploaded documents
- Memory-augmented chat with LLMs
- Runs fully locally (via Ollama)

---

## Steps to get started

### 1. Clone the repository

```bash
git clone https://github.com/Saarth236/RAG_Pipeline.git
cd RAG_Pipeline

### 2. Add Your Documents

Place `.pdf`, `.txt`, or `.csv` files inside the `docs/` folder.

Then run:

```bash
python ingest_new_docs.py

