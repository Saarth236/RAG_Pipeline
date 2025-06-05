import random

with open("chunk_texts.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()  # Read entire file
    all_chunks = raw_text.strip().split("\n\n")  # Split based on double newlines

# Select 5 random chunks
random_chunks = random.sample(all_chunks, min(5, len(all_chunks)))

print("\n🔍 Displaying Random Chunks from chunk_texts.txt:\n")
for i, chunk in enumerate(random_chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n" + "-" * 80)
