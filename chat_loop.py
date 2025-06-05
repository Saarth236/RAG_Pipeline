import os
import faiss
import json
import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer

index_file = "faiss.index"
chunk_file = "chunk_texts.txt"
chat_memory_file = "chat_memory.txt"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_index():
    if os.path.exists(index_file):
        return faiss.read_index(index_file)
    print("❌ FAISS index not found! Please run ingest_new_docs.py first.")
    exit()

def load_chunks():
    if os.path.exists(chunk_file):
        with open(chunk_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def load_memory():
    if os.path.exists(chat_memory_file):
        with open(chat_memory_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def save_memory(user_query, response):
    with open(chat_memory_file, "a", encoding="utf-8") as f:
        f.write(f"User: {user_query}\nBot: {response}\n\n")

def retrieve(query, k=5):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i in indices[0]:
        if 0 <= i < len(all_chunks):
            results.append(all_chunks[i])
    
    return results

def build_prompt(memory, chunks, query):
    memory_text = "\n".join(memory[-5:]) 
    context = "\n\n".join(chunks)
    
    return f"""
You are a helpful assistant. You have access to the following past conversation:

{memory_text}

Additionally, you have access to the following context:

{context}

Answer the user's question. If the context does not have the answer, say so.

User question: {query}
"""

def stream_response(prompt):
    payload = {"model": "Deepseek-R1:8B", "prompt": prompt}
    r = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    full_response = ""
    previous_word = ""

    print("\n📝 Response:", end=" ", flush=True)

    for line in r.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            response_text = data.get("response", "").strip()


            if previous_word and not previous_word.endswith((" ", ".", ",", "!", "?", "-")):
                response_text = " " + response_text 

            response_text = re.sub(r"<think>", "\n<think>\n", response_text)
            response_text = re.sub(r"</think>", "\n</think>\n", response_text)

            print(response_text, end="", flush=True)
            full_response += response_text
            previous_word = response_text  

            if data.get("done"):
                break

    print("\n") 
    return full_response.strip()

index = load_index()
all_chunks = load_chunks()
conversation_memory = load_memory()

def chat_loop():
    if index.ntotal == 0:
        print("❌ FAISS index is empty! Please run ingest_new_docs.py first.")
        return
    
    print("\n💬 Chatbot Ready! Type 'exit' to quit.")
    
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == "exit":
            break
        
        top_chunks = retrieve(user_query, k=5)
        if not top_chunks:
            print("\n❌ No relevant information found.\n")
            continue

        print("\n🔍 Retrieved Chunks:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"\nChunk {i}:\n{chunk[:500]}...\n")  

        prompt_text = build_prompt(conversation_memory, top_chunks, user_query)
        response = stream_response(prompt_text)
        save_memory(user_query, response)
        conversation_memory.append(f"User: {user_query}")
        conversation_memory.append(f"Bot: {response}")

if __name__ == "__main__":
    chat_loop()
