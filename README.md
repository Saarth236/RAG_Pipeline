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

# 1. Clone the repository

git clone https://github.com/Saarth236/RAG_Pipeline.git
cd RAG_Pipeline

# 2. Add Your Documents

Place `.pdf`, `.txt`, or `.csv` files inside the `docs/` folder

# 3. Create a virtual environment named 'venv' and activate it
python3 -m venv venv
source venv/bin/activate

# 4. Install the necessary python libraries through the requirements txt file
pip install -r requirements.txt

# 5. INstall Ollama and use the recommended Deepseek model for usage
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek:8b
 
# And finally, run the code to ingest docs 
python ingest_new_docs.py

python chat_loop.py


