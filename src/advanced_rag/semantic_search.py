import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure src/ is in the python path to import helpers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

DATA_DIR = os.path.join(BASE_DIR, "data")
from helpers.llm_helper import call_llm_with_full_text, print_formatted_response

# A small, fast, well-balanced embedding model. Downloaded once (~90MB) and cached.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_documents(data_dir):
    """Load all .txt files from the data directory."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            documents.append({"name": filename, "content": content})
    return documents

def chunk_documents(documents, chunk_size=50):
    """Split each document into word-based chunks, keeping the source filename."""
    chunks = []
    for doc in documents:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({"text": chunk_text, "source": doc["name"]})
    return chunks

def build_embedding_index(records, model):
    """Embed every chunk into a dense vector once (the 'index').

    normalize_embeddings=True makes each vector unit-length, so a plain dot
    product equals cosine similarity — no separate normalization step needed.
    """
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings

def find_top_matches(query, model, embeddings, top_k=3, min_score=0.2):
    """Semantic retrieval: rank chunks by cosine similarity to the query.

    Unlike TF-IDF, this compares *meaning*, so a relevant chunk scores high even
    with no shared words. Returns a list of (score, index) above min_score.
    """
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ q_emb  # cosine (vectors are unit-length)

    ranked = np.argsort(similarities)[::-1][:top_k]
    return [(float(similarities[i]), int(i)) for i in ranked if similarities[i] >= min_score]

if __name__ == "__main__":
    print("=" * 60)
    print("  Advanced RAG - Semantic Search (Sentence Embeddings)")
    print("=" * 60)

    print("\n[1] Loading knowledge base...")
    documents = load_documents(DATA_DIR)
    db_records = chunk_documents(documents, chunk_size=50)

    print(f"\n[2] Loading embedding model '{EMBEDDING_MODEL}' and building index...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = build_embedding_index(db_records, model)
    print(f"    Index built! {embeddings.shape[0]} chunks x {embeddings.shape[1]} dimensions.")

    query = "define a rag store"
    print(f"\n[3] User Query: \"{query}\"\n")
    print("    Finding the best matching documents using Semantic Search...\n")

    top_matches = find_top_matches(query, model, embeddings, top_k=3)

    if not top_matches:
        print("    No relevant match found above the similarity threshold.")
        sys.exit(0)

    # The first entry is the highest-ranked (best) match.
    best_score, best_index = top_matches[0]
    best_record = db_records[best_index]
    print(f"    Best Semantic Match Score: {best_score:.3f}")
    print(f"    Match Text: ({best_record['source']}) {best_record['text'][:100]}...\n")

    print(f"    Top {len(top_matches)} matches:")
    for rank, (score, idx) in enumerate(top_matches, start=1):
        record = db_records[idx]
        print(f"    {rank}. [{score:.3f}] ({record['source']}) {record['text'][:80]}...")
    print()

    # Augment the query with the ranked, cited context
    print("[4] Augmenting the user query with retrieved context...")
    context_blocks = [
        f"[{db_records[idx]['source']}] {db_records[idx]['text']}"
        for _, idx in top_matches
    ]
    augmented_input = query + ":\n" + "\n\n".join(context_blocks)
    print_formatted_response(augmented_input)

    # Generation with Gemini
    print("\n[5] Generation with Gemini...\n")
    llm_response = call_llm_with_full_text([augmented_input])
    print_formatted_response(llm_response)
