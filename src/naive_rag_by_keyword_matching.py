# Advanced techniques and evaluation
import os
import re
from helpers.llm_helper import call_llm_with_full_text, print_formatted_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


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
            chunks.append({
                "source": doc["name"],
                "text": chunk_text,
            })
    return chunks


# ──────────────────────────────────────────────
# 3. Keyword Extraction
# ──────────────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "is", "are", "of", "it", "this", "that", "by", "from", "as",
    "be", "was", "were", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "no", "so", "if", "than", "too", "very", "just", "about", "into",
}

def extract_keywords(text):
    """Tokenize, lowercase, remove punctuation & stop words."""
    clean = re.sub(r'[^\w\s]', '', text.lower())
    return {w for w in clean.split() if w not in STOP_WORDS and len(w) > 2}


# ──────────────────────────────────────────────
# 4. Keyword-Matching Retrieval
# ──────────────────────────────────────────────
def retrieve_best_chunks(query, chunks, top_k=2):
    """Return the chunk(s) with the highest keyword overlap."""
    query_kw = extract_keywords(query)
    scored = []
    for chunk in chunks:
        # compute the number of words shared between each chunk and query
        overlap = len(query_kw & extract_keywords(chunk["text"]))
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ──────────────────────────────────────────────
# 5. Run the Naive RAG pipeline
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Naive RAG - Keyword Matching")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1] Loading knowledge base...\n")
    documents = load_documents(DATA_DIR)
    print(f"\n    Total documents loaded: {len(documents)}")

    # Step 2: Chunk
    print("\n[2] Chunking documents (200 words per chunk)...\n")
    chunks = chunk_documents(documents, chunk_size=200)
    print(f"    Total chunks created: {len(chunks)}")

    # Step 3: Query & Retrieve
    query = "What is Advanced RAG and how does it improve retrieval?"
    print(f"\n[3] User Query: \"{query}\"\n")
    print("    Retrieving best chunks using Keyword Matching...\n")

    results = retrieve_best_chunks(query, chunks, top_k=2)

    for rank, (score, chunk) in enumerate(results, 1):
        print(f"    #{rank}  score={score}  source={chunk['source']}")
        print(f"        preview: {chunk['text'][:100].strip()}...\n")

    # Step 4: Augment & Generate
    best_context = "\n\n".join(chunk["text"] for _, chunk in results)

    print("[4] Sending augmented request to Gemini...\n")
    itext = [
        "Context:",
        best_context,
        "Question:",
        query,
    ]
    response = call_llm_with_full_text(itext)
    print_formatted_response(response)
