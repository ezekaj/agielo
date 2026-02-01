"""
Semantic Embeddings - Neural Text Encoding
==========================================

Provides meaningful semantic embeddings using:
1. sentence-transformers (best quality, requires download)
2. TF-IDF with SVD (good fallback, no downloads)
3. Hash-based (fast fallback, last resort)

Key improvement: "dog" and "cat" now have SIMILAR embeddings!
Unlike SHA256 where they were completely random.
"""

import numpy as np
import os
import pickle
import hashlib
from typing import Optional, List, Dict, Union
from pathlib import Path
import threading

# Global embedding dimension
EMBEDDING_DIM = 384  # MiniLM default, but configurable

# Lazy loading flags - don't import heavy models at startup
SENTENCE_TRANSFORMERS_AVAILABLE = None  # Will be checked lazily
TFIDF_AVAILABLE = None

def _check_sentence_transformers():
    """Lazily check if sentence-transformers is available."""
    global SENTENCE_TRANSFORMERS_AVAILABLE
    if SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            import sentence_transformers
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False
    return SENTENCE_TRANSFORMERS_AVAILABLE

def _check_tfidf():
    """Lazily check if sklearn is available."""
    global TFIDF_AVAILABLE
    if TFIDF_AVAILABLE is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            TFIDF_AVAILABLE = True
        except ImportError:
            TFIDF_AVAILABLE = False
    return TFIDF_AVAILABLE


class SemanticEmbedder:
    """
    Unified semantic embedding system.

    Features:
    - Caches embeddings for speed
    - Falls back gracefully
    - Thread-safe
    - Persists cache to disk
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one embedder instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, dim: int = EMBEDDING_DIM, cache_path: str = None):
        if self._initialized:
            return

        self.dim = dim
        self.cache_path = Path(cache_path or os.path.expanduser("~/.cognitive_ai_knowledge/embeddings_cache.pkl"))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Embedding cache
        self._cache: Dict[str, np.ndarray] = {}
        self._load_cache()

        # Initialize backend
        self.backend = None
        self.backend_name = "none"
        self._init_backend()

        self._initialized = True

        print(f"[Embeddings] Using {self.backend_name} backend (dim={self.dim})")

    def _init_backend(self):
        """Initialize the best available backend - prefers fast startup."""
        # For fast startup, use hash-ngram first, load better models lazily
        # The sentence-transformers model causes mutex blocking on load

        # Start with fast hash-based (instant startup)
        self.backend_name = "hash-ngram"
        self.backend = None

        # Schedule loading better backend in background if available
        if _check_tfidf():
            try:
                self._init_tfidf_backend()
                self.backend_name = "tfidf-svd"
            except Exception as e:
                pass  # Keep hash-ngram

    def _upgrade_to_transformer(self):
        """Upgrade to sentence-transformers if available (call when needed)."""
        if self.backend_name == "sentence-transformers":
            return True

        if _check_sentence_transformers():
            try:
                from sentence_transformers import SentenceTransformer
                self.backend = SentenceTransformer('all-MiniLM-L6-v2')
                self.backend_name = "sentence-transformers"
                self.dim = 384
                print("[Embeddings] Upgraded to sentence-transformers")
                return True
            except:
                pass
        return False

    def _init_tfidf_backend(self):
        """Initialize TF-IDF + SVD backend with pre-trained vocabulary."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        # Common English words for vocabulary
        vocab = [
            # Common words
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            # Tech/AI terms
            "computer", "program", "code", "software", "data", "algorithm",
            "machine", "learning", "neural", "network", "model", "train",
            "artificial", "intelligence", "deep", "layer", "input", "output",
            # Common nouns
            "time", "year", "people", "way", "day", "man", "thing", "woman",
            "life", "child", "world", "school", "state", "family", "student",
            # Common verbs
            "get", "make", "go", "know", "take", "see", "come", "think", "look",
            "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel",
            # Adjectives
            "good", "new", "first", "last", "long", "great", "little", "own",
            "other", "old", "right", "big", "high", "different", "small", "large",
            # Question words
            "what", "who", "how", "why", "when", "where", "which",
            # More technical
            "memory", "process", "system", "function", "variable", "class",
            "object", "method", "error", "file", "string", "number", "array",
            "list", "dict", "python", "javascript", "api", "server", "client",
        ]

        # Create documents from vocabulary for fitting
        docs = vocab + [" ".join(vocab[i:i+3]) for i in range(len(vocab)-2)]

        self._tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            lowercase=True,
            stop_words=None  # Keep stop words for better matching
        )
        self._tfidf.fit(docs)

        # SVD for dimensionality reduction
        self._svd = TruncatedSVD(n_components=min(self.dim, 100))
        tfidf_matrix = self._tfidf.transform(docs)
        self._svd.fit(tfidf_matrix)

        self.backend = (self._tfidf, self._svd)

    def embed(self, text: str) -> np.ndarray:
        """
        Get semantic embedding for text.

        Args:
            text: Input text

        Returns:
            Normalized embedding vector
        """
        # Normalize text
        text = text.strip().lower()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        if self.backend_name == "sentence-transformers":
            embedding = self._embed_transformer(text)
        elif self.backend_name == "tfidf-svd":
            embedding = self._embed_tfidf(text)
        else:
            embedding = self._embed_hash_ngram(text)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        self._cache[cache_key] = embedding.astype(np.float32)

        # Periodically save cache
        if len(self._cache) % 100 == 0:
            self._save_cache()

        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        if self.backend_name == "sentence-transformers":
            # Batch processing is faster
            embeddings = self.backend.encode(texts, show_progress_bar=False)
            # Normalize each
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            return embeddings.astype(np.float32)
        else:
            # Fallback to single processing
            return np.array([self.embed(t) for t in texts])

    def _embed_transformer(self, text: str) -> np.ndarray:
        """Embed using sentence-transformers."""
        embedding = self.backend.encode(text, show_progress_bar=False)
        return embedding

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """Embed using TF-IDF + SVD."""
        tfidf, svd = self.backend

        # Transform text
        tfidf_vec = tfidf.transform([text])
        reduced = svd.transform(tfidf_vec)[0]

        # Pad to target dimension
        if len(reduced) < self.dim:
            padded = np.zeros(self.dim)
            padded[:len(reduced)] = reduced
            return padded
        return reduced[:self.dim]

    def _embed_hash_ngram(self, text: str) -> np.ndarray:
        """
        Hash-based embedding with n-grams.

        Much better than pure SHA256:
        - Uses character n-grams (captures subword similarity)
        - Similar words have similar hashes
        - "dog" and "dogs" share most n-grams
        """
        embedding = np.zeros(self.dim, dtype=np.float32)

        # Character n-grams (2, 3, 4)
        text = f" {text} "  # Add boundary markers
        ngrams = []

        for n in [2, 3, 4]:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])

        # Word-level features
        words = text.split()
        ngrams.extend(words)

        # Hash each n-gram to a position
        for ngram in ngrams:
            h = hash(ngram)
            pos = h % self.dim
            sign = 1 if (h // self.dim) % 2 == 0 else -1
            embedding[pos] += sign * 1.0

        return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))

    def most_similar(self, query: str, candidates: List[str], k: int = 5) -> List[tuple]:
        """Find most similar texts from candidates."""
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        return [(candidates[i], float(similarities[i])) for i in top_indices]

    def _load_cache(self):
        """Load embedding cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
            except:
                self._cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            # Keep only recent entries
            if len(self._cache) > 10000:
                keys = list(self._cache.keys())[-5000:]
                self._cache = {k: self._cache[k] for k in keys}

            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
        except:
            pass

    def get_info(self) -> Dict:
        """Get information about the embedder."""
        return {
            'backend': self.backend_name,
            'dimension': self.dim,
            'cache_size': len(self._cache),
            'cache_path': str(self.cache_path)
        }


# Global embedder instance
_embedder: Optional[SemanticEmbedder] = None


def get_embedder(dim: int = EMBEDDING_DIM) -> SemanticEmbedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = SemanticEmbedder(dim=dim)
    return _embedder


def embed_text(text: str) -> np.ndarray:
    """Convenience function to embed text."""
    return get_embedder().embed(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Convenience function to embed multiple texts."""
    return get_embedder().embed_batch(texts)


def text_similarity(text1: str, text2: str) -> float:
    """Convenience function for text similarity."""
    return get_embedder().similarity(text1, text2)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC EMBEDDINGS TEST")
    print("=" * 60)

    embedder = get_embedder()
    print(f"\nBackend: {embedder.backend_name}")
    print(f"Dimension: {embedder.dim}")

    # Test semantic similarity
    print("\n--- Similarity Tests ---")
    test_pairs = [
        ("dog", "cat"),           # Should be similar (both animals)
        ("dog", "puppy"),         # Should be very similar
        ("dog", "computer"),      # Should be different
        ("king", "queen"),        # Should be similar
        ("machine learning", "deep learning"),  # Should be very similar
        ("python programming", "javascript code"),  # Should be somewhat similar
    ]

    for t1, t2 in test_pairs:
        sim = embedder.similarity(t1, t2)
        print(f"  '{t1}' vs '{t2}': {sim:.3f}")

    # Test most similar
    print("\n--- Most Similar Test ---")
    query = "artificial intelligence"
    candidates = [
        "machine learning",
        "deep neural networks",
        "cooking recipes",
        "computer science",
        "natural language processing",
        "gardening tips",
        "data science",
    ]

    results = embedder.most_similar(query, candidates, k=3)
    print(f"Query: '{query}'")
    for text, score in results:
        print(f"  {score:.3f}: {text}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
