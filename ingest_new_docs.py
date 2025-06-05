import os
import fitz  # PyMuPDF for PDFs
import pandas as pd
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 📂 Paths
docs_folder = "docs"
processed_list_file = "processed_files.txt"
chunk_file = "chunk_texts.txt"
index_file = "faiss.index"

# 🧠 Load Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🏗 Load FAISS index
def load_index():
    if os.path.exists(index_file):
        return faiss.read_index(index_file)
    d = model.get_sentence_embedding_dimension()
    return faiss.IndexFlatL2(d)

# 📖 Extract text
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n\n".join([page.get_text("text").strip() for page in doc if page.get_text("text").strip()])
    elif file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        return " ".join(df.astype(str).values.flatten())
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# 🔄 Load processed files
if not os.path.exists(processed_list_file):
    open(processed_list_file, "w", encoding="utf-8").close()
if not os.path.exists(chunk_file):
    open(chunk_file, "w", encoding="utf-8").close()

with open(processed_list_file, "r", encoding="utf-8") as f:
    processed_files = set(x.strip() for x in f if x.strip())

index = load_index()

# 📂 Process new documents
for fn in os.listdir(docs_folder):
    full_path = os.path.join(docs_folder, fn)

    if os.path.isfile(full_path) and fn not in processed_files:
        text = extract_text(full_path)

        if text:
            print(f"📄 Processing new document: {fn}")

            # 🚀 Use **Larger overlapping chunks**
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            chunks = splitter.split_text(text)
            embeddings = model.encode(chunks).astype("float32")

            # 🏗 Add embeddings to FAISS
            index.add(embeddings)
            faiss.write_index(index, index_file)

            # 📂 Save chunked text for retrieval
            with open(chunk_file, "a", encoding="utf-8") as cfile:
                for c in chunks:
                    cfile.write(c.replace("\n", " ").strip() + "\n\n")  # Ensure each chunk is a single paragraph


            # ✅ Move file to processed
            processed_files.add(fn)

            # 🔍 Debug: Show first few extracted chunks
            print(f"🔍 First few chunks from {fn}:")
            for i, c in enumerate(chunks[:5]):
                print(f"\nChunk {i+1}:\n{c}\n")

# 🔄 Save processed files list
with open(processed_list_file, "w", encoding="utf-8") as w:
    for p in processed_files:
        w.write(p + "\n")

print(f"✅ FAISS now contains {index.ntotal} document vectors!")
