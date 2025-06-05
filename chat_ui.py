import os
import requests
import json
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# File paths for memory persistence
memory_index_file = "chat_memory.faiss"
memory_text_file = "chat_memory.txt"

# Load FAISS memory if it exists
if os.path.exists(memory_index_file):
    memory_index = faiss.read_index(memory_index_file)
else:
    memory_index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

# Load stored chat history (text-based)
if os.path.exists(memory_text_file):
    with open(memory_text_file, "r", encoding="utf-8") as f:
        all_memories = [line.strip() for line in f.readlines() if line.strip()]
else:
    all_memories = []

def store_in_memory(query, response):
    """Store chat history into FAISS and a text file."""
    combined_text = f"User: {query}\nBot: {response}"
    
    # Convert to embedding & store in FAISS
    embedding = model.encode([combined_text]).astype("float32")
    memory_index.add(embedding)
    
    # Append to text memory & save
    all_memories.append(combined_text)
    with open(memory_text_file, "a", encoding="utf-8") as f:
        f.write(combined_text + "\n\n")

    # Save FAISS index
    faiss.write_index(memory_index, memory_index_file)

def retrieve_memory(query, k=3):
    """Retrieve past user queries & bot responses from FAISS memory."""
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = memory_index.search(query_embedding, k)

    retrieved_memories = []
    for i in indices[0]:
        if 0 <= i < len(all_memories):
            retrieved_memories.append(all_memories[i])

    return retrieved_memories

def retrieve(query, k=5):
    """Retrieve relevant document chunks from FAISS."""
    try:
        index = faiss.read_index("faiss.index")
        with open("chunk_texts.txt", "r", encoding="utf-8") as f:
            all_chunks = [x.strip() for x in f.readlines()]
    except:
        return ["No document FAISS index found. Run `ingest_new_docs.py` first."]

    qe = model.encode([query])
    distances, indices = index.search(qe, k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(all_chunks):
            results.append(all_chunks[i])

    return results

def chatbot_response(user_query):
    """Process user input, retrieve relevant data, and generate response."""
    # Retrieve document-based knowledge
    top_chunks = retrieve(user_query, 5)
    document_context = "\n".join(top_chunks)

    # Retrieve long-term chat memory
    past_memories = retrieve_memory(user_query, 3)
    memory_context = "\n".join(past_memories)

    # Build the final prompt
    prompt_text = f"""
You are a helpful AI assistant. Here is the conversation so far:

Past Memory:
{memory_context}

Retrieved Document Context:
{document_context}

New User Query: {user_query}

If none of the information is relevant, say so.
"""

    # Send to DeepSeek API (Ollama)
    payload = {"model": "Deepseek-R1:8B", "prompt": prompt_text}
    r = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    # Read API response
    full_response = ""
    for line in r.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            full_response += data["response"]
            if data["done"]:
                break

    # Store response in memory
    store_in_memory(user_query, full_response)

    return full_response

# Gradio UI
with gr.Blocks() as chat_ui:
    gr.Markdown("# 🤖 AI Chatbot with Memory & Document Retrieval")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Ask a question:")
    send_button = gr.Button("Send")

    def respond(history, user_message):
        response = chatbot_response(user_message)
        history.append((user_message, response))
        return history, ""

    send_button.click(respond, inputs=[chatbot, user_input], outputs=[chatbot, user_input])

# Launch Gradio UI
chat_ui.launch(server_name="0.0.0.0", server_port=7860)
