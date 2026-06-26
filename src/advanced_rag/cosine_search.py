import os
import sys

# Ensure src/ is in the python path to import helpers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

DATA_DIR = os.path.join(BASE_DIR, "data")

import re
import math
from collections import Counter
from helpers.llm_helper import call_llm_with_full_text, print_formatted_response
from helpers.constants import STOP_WORDS



# ──────────────────────────────────────────────
# 1. Load Knowledge Base from text files
# ──────────────────────────────────────────────
def load_documents(data_dir):
    """Load all .txt files from the data directory."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append({"name": filename, "content": content})
            print(f"  Loaded: {filename} ({len(content.split())} words)")
    return documents

# ──────────────────────────────────────────────
# 2. Chunking
# ──────────────────────────────────────────────
def chunk_documents(documents, chunk_size=200):
    """Split each document into word-based chunks."""
    chunks = []
    for doc in documents:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append(chunk_text)
    return chunks

# ──────────────────────────────────────────────
# 3. Vectorization and Similarity (Pure Python)
# ──────────────────────────────────────────────
def get_words(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def calculate_cosine_similarity(text1, text2):
    """Simple term-frequency based cosine similarity (like basic vector search)."""
    words1 = get_words(text1)
    words2 = get_words(text2)
    
    # unique vocabs of each text
    vocab = list(set(words1) | set(words2))
    if not vocab:
        return 0.0
        
    c1 = Counter(words1)
    c2 = Counter(words2)
    
    v1 = [c1[w] for w in vocab] # vector of text1 (user query)
    v2 = [c2[w] for w in vocab] # vector of tex2 (db record)
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

def find_best_match(text_input, records):
    """Find the best match using cosine similarity over vectors."""
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

# ──────────────────────────────────────────────
# 5. Advanced RAG with Vectors Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Advanced RAG - Vector Search (Cosine Similarity)")
    print("=" * 60)

    # 1. Load Data
    print("\n[1] Loading knowledge base...\n")
    documents = load_documents(DATA_DIR)
    db_records = chunk_documents(documents, chunk_size=50) # Using smaller chunks for sentences

    # 2. Vector Search (Loop-based Cosine Similarity)
    query = "define a rag store"
    print(f"\n[2] User Query: \"{query}\"\n")
    print("    Finding the best matching document using Vector Search (Loop Cosine Similarity)...\n")

    best_similarity_score, best_matching_record = find_best_match(query, db_records)

    if best_matching_record:
        print_formatted_response(best_matching_record)
        print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")


        # Augmented Input
        print("\n[4] Augmenting the user query...")
        augmented_input = query + ": " + best_matching_record
        print_formatted_response(augmented_input)

        # LLM Generation
        print("\n[5] Generation with Gemini...\n")
        llm_response = call_llm_with_full_text([augmented_input])
        print_formatted_response(llm_response)
    else:
        print("No match found in the dataset.")
