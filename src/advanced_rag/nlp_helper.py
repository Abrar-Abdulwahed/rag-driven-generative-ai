import spacy
import nltk
from nltk.corpus import wordnet
import numpy as np

# Load SpaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    raise

def get_synonyms(word):
    """Retrieves synonyms for a given word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Add synonym and replace underscores with spaces
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def preprocess_text(text):
    """Converts text to lowercase, lemmatizes, and filters stopwords and punctuation."""
    doc = nlp(text.lower())
    processed_words = []
    for token in doc:
        # Keep non-stopword, non-punctuation alphabetic words
        if not token.is_stop and not token.is_punct and token.is_alpha:
            processed_words.append(token.lemma_)
    return processed_words

def expand_with_synonyms(words):
    """Enhances a list of words by adding their synonyms."""
    expanded_set = set(words)
    for word in words:
        syns = get_synonyms(word)
        expanded_set.update(syns)
    return list(expanded_set)

def calculate_enhanced_similarity(text1, text2):
    """Computes cosine similarity between preprocessed and synonym-expanded text vectors."""
    # 1. Preprocess texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    if not words1 or not words2:
        return 0.0
        
    # 2. Expand with synonyms
    expanded1 = expand_with_synonyms(words1)
    expanded2 = expand_with_synonyms(words2)
    
    # 3. Create vocabulary
    vocab = list(set(expanded1) | set(expanded2))
    
    # 4. Create vectors
    v1 = np.array([1 if word in expanded1 else 0 for word in vocab])
    v2 = np.array([1 if word in expanded2 else 0 for word in vocab])
    
    # 5. Compute cosine similarity using numpy
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)
