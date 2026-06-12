import os
import sys

# Ensure src/ is in the python path to import helpers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
DATA_DIR = os.path.join(BASE_DIR, "data")

from helpers.llm_helper import call_llm_with_full_text, print_formatted_response
from advanced_rag.nlp_helper import calculate_enhanced_similarity

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
    """Split each document into word-based chunks."""
    chunks = []
    for doc in documents:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append(chunk_text)
    return chunks

def find_best_semantic_match(text_input, records):
    """Find the best match using enhanced semantic similarity."""
    best_score = 0
    best_record = None
    
    for i, record in enumerate(records):
        current_score = calculate_enhanced_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
            
    return best_score, best_record

if __name__ == "__main__":
    print("=" * 60)
    print("  Advanced RAG - Enhanced Semantic Similarity (spaCy + NLTK)")
    print("=" * 60)

    print("\n[1] Loading knowledge base...")
    documents = load_documents(DATA_DIR)
    db_records = chunk_documents(documents, chunk_size=50)

    query = "define a rag store"
    print(f"\n[2] User Query: \"{query}\"\n")
    print("    Finding the best matching document using Enhanced Semantic Similarity...")
    print("    (This analyzes synonyms and word meanings, which takes slightly longer...)\n")

    best_score, best_matching_record = find_best_semantic_match(query, db_records)

    if best_matching_record:
        print(f"    Best Enhanced Similarity Score: {best_score:.3f}")
        print(f"    Match Text: {best_matching_record[:100]}...")

        print("\n[3] Augmenting the user query...")
        augmented_input = query + ": " + best_matching_record
        print_formatted_response(augmented_input)

        print("\n[4] Generation with Gemini...\n")
        llm_response = call_llm_with_full_text([augmented_input])
        print_formatted_response(llm_response)
    else:
        print("No match found in the dataset.")
