import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure src/ is in the python path to import helpers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

DATA_DIR = os.path.join(BASE_DIR, "data")
from helpers.llm_helper import call_llm_with_full_text, print_formatted_response

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

def setup_vectorizer(records):
    """Sets up the Scikit-Learn TF-IDF vectorizer and builds the matrix index."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix

def find_best_match(query, vectorizer, tfidf_matrix):
    """Finds the best match using the query against the pre-indexed matrix."""
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()
    best_score = similarities[0, best_index]
    return best_score, best_index

if __name__ == "__main__":
    print("=" * 60)
    print("  Advanced RAG - Scikit-Learn Index-Based Search")
    print("=" * 60)

    print("\n[1] Loading knowledge base...")
    documents = load_documents(DATA_DIR)
    db_records = chunk_documents(documents, chunk_size=50)

    query = "define a rag store"
    print(f"\n[2] User Query: \"{query}\"\n")
    print("    Finding the best matching document using Index-Based Search (Scikit-Learn)...\n")

    # 1. Build Index
    vectorizer, tfidf_matrix = setup_vectorizer(db_records)
    
    # Feature Extraction verification
    feature_names = vectorizer.get_feature_names_out()
    print("    Feature Extraction Check:")
    print("    First 5 features in vocabulary:", list(feature_names[:5]))
    print(f"    Total Document Matrix Shape: {tfidf_matrix.shape}\n")

    # 2. Search Index
    best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
    best_matching_record = db_records[best_index]

    print(f"    Best Index Match Score: {best_similarity_score:.3f}")
    print(f"    Match Text: {best_matching_record[:100]}...\n")

    print("[3] Augmenting the user query...")
    augmented_input = query + ": " + best_matching_record
    print_formatted_response(augmented_input)

    print("\n[4] Generation with Gemini...\n")
    llm_response = call_llm_with_full_text([augmented_input])
    print_formatted_response(llm_response)
