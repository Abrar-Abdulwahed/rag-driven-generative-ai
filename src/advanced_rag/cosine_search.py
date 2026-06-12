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

def calculate_enhanced_similarity(text1, text2):
    """Enhanced TF-IDF style similarity (simulating the higher score)."""
    # This is a simplified proxy for a more complex TF-IDF or embedding
    # We apply a boost to emphasize matching meaningful words.
    words1 = set(w for w in get_words(text1) if len(w) > 2)
    words2 = set(w for w in get_words(text2) if len(w) > 2)
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1 & words2
    # Boost the score based on overlap of significant words
    score = len(intersection) / math.sqrt(len(words1) * len(words2))
    return score

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
# 4. Index-Based Search (Pre-computed Matrix)
# ──────────────────────────────────────────────
def build_tfidf_index(records):
    """Build a global vocabulary, IDF, and TF-IDF matrix for all records."""
    df = Counter()
    total_docs = len(records)
    doc_words = []
    
    for record in records:
        words = get_words(record)
        doc_words.append(words)
        for word in set(words):
            df[word] += 1
            
    vocab = list(df.keys())
    
    idf = {}
    for word in vocab:
        idf[word] = math.log(total_docs / (df[word] + 1)) + 1
        
    tfidf_matrix = []
    for words in doc_words:
        tf = Counter(words)
        doc_len = len(words) if words else 1
        vector = []
        for word in vocab:
            term_freq = tf[word] / doc_len
            vector.append(term_freq * idf[word])
        tfidf_matrix.append(vector)
        
    return vocab, idf, tfidf_matrix

def index_based_search(query, tfidf_matrix, vocab, idf, records):
    """Search using a pre-computed TF-IDF index matrix."""
    query_words = get_words(query)
    q_tf = Counter(query_words)
    q_len = len(query_words) if query_words else 1
    
    q_vec = []
    for word in vocab:
        term_freq = q_tf[word] / q_len
        q_vec.append(term_freq * idf[word])
        
    best_score = 0
    best_index = -1
    
    q_norm = math.sqrt(sum(a * a for a in q_vec))
    if q_norm == 0:
        return 0.0, None
        
    for i, doc_vec in enumerate(tfidf_matrix):
        dot_product = sum(a * b for a, b in zip(q_vec, doc_vec))
        d_norm = math.sqrt(sum(b * b for b in doc_vec))
        
        if d_norm == 0:
            continue
            
        sim = dot_product / (q_norm * d_norm)
        if sim > best_score:
            best_score = sim
            best_index = i
            
    if best_index != -1:
        return best_score, records[best_index]
    return 0.0, None

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

        # Try enhanced similarity
        enhanced_score = calculate_enhanced_similarity(query, best_matching_record)
        print(f"\n    {query} : {best_matching_record[:100]}...")
        print(f"    Enhanced Similarity: {enhanced_score:.3f}")

        # 3. Index-Based Search (Pre-computed TF-IDF Matrix)
        print("\n[3] Building TF-IDF Index Matrix...")
        vocab, idf, tfidf_matrix = build_tfidf_index(db_records)
        print(f"    Index built! Matrix dimensions: {len(tfidf_matrix)} docs x {len(vocab)} words.")
        
        print("\n    Finding the best matching document using Index-Based Search...")
        idx_score, idx_record = index_based_search(query, tfidf_matrix, vocab, idf, db_records)
        
        if idx_record:
            print(f"    Best Index Match Score: {idx_score:.3f}")
            print(f"    Match Text: {idx_record[:100]}...")

        # 4. Augmented Input
        print("\n[4] Augmenting the user query...")
        augmented_input = query + ": " + best_matching_record
        print_formatted_response(augmented_input)

        # 5. LLM Generation
        print("\n[5] Generation with Gemini...\n")
        llm_response = call_llm_with_full_text([augmented_input])
        print_formatted_response(llm_response)
    else:
        print("No match found in the dataset.")
