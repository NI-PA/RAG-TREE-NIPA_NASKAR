"""
vector_store.py
---------------
VectorStore class with custom cosine similarity, add(), and query() methods.
Uses TfidfVectorizer from scikit-learn for vectorization.
Supports dynamic document ingestion from PDF files.
"""

from __future__ import annotations
import numpy as np
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorStore:
    """
    A simple in-memory vector store that uses TF-IDF embeddings
    and custom cosine similarity for retrieval.
    Supports dynamic loading of content from PDF files.
    """

    def __init__(self, name: str):
        """
        Initialize the vector store with a name.

        Args:
            name: A descriptive name for this store.
        """
        self.name = name
        self._entries: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self._is_fitted = False

    def _rebuild_vectors(self) -> None:
        """Rebuild all vectors by re-fitting TF-IDF on all stored texts."""
        if not self._entries:
            self._vectors = []
            self._is_fitted = False
            return

        all_texts = [entry['text'] for entry in self._entries]
        tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        self._vectors = [
            np.array(tfidf_matrix[i].toarray()[0], dtype=np.float64)
            for i in range(tfidf_matrix.shape[0])
        ]
        self._is_fitted = True

    def _vectorize(self, text: str) -> np.ndarray:
        """Convert text to a vector using the fitted TF-IDF vectorizer."""
        if not self._is_fitted:
            return np.zeros(1, dtype=np.float64)
        vec = self._vectorizer.transform([text])
        return np.array(vec.toarray()[0], dtype=np.float64)

    def add(self, text: str, payload: dict) -> None:
        """Add text and its payload to the store, then rebuild all vectors."""
        self._entries.append({
            'text': text,
            'payload': payload
        })
        self._rebuild_vectors()

    def add_bulk(self, items: list[dict]) -> None:
        """
        Add multiple items at once (more efficient than calling add() in a loop).
        
        Args:
            items: List of dicts, each with 'text' and 'payload' keys.
        """
        for item in items:
            self._entries.append({
                'text': item['text'],
                'payload': item['payload']
            })
        self._rebuild_vectors()

    def clear(self) -> None:
        """Clear all entries from the store."""
        self._entries = []
        self._vectors = []
        self._is_fitted = False

    def query(self, text: str, k: int) -> list[dict]:
        """Return top-k results sorted by cosine similarity (descending)."""
        if not self._entries:
            return []

        query_vec = self._vectorize(text)
        scored = []
        for i, entry in enumerate(self._entries):
            score = VectorStore.cosine_similarity(query_vec.tolist(), self._vectors[i].tolist())
            scored.append({
                'payload': entry['payload'],
                'score': score,
                'text': entry['text'],
                'vector': self._vectors[i]
            })
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:k]

    def query_all_scored(self, query_vec: np.ndarray) -> list[dict]:
        """Return ALL entries scored against the provided query vector, sorted descending."""
        scored = []
        for i, entry in enumerate(self._entries):
            score = VectorStore.cosine_similarity(query_vec.tolist(), self._vectors[i].tolist())
            scored.append({
                'payload': entry['payload'],
                'score': score,
                'text': entry['text'],
                'vector': self._vectors[i]
            })
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored

    def get_all_vectors(self) -> list[np.ndarray]:
        """Return all stored vectors."""
        return list(self._vectors)

    def size(self) -> int:
        """Return number of entries in the store."""
        return len(self._entries)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors from scratch.
        Formula: cos(a, b) = (a . b) / (||a|| * ||b||)
        """
        if len(a) != len(b):
            raise ValueError(
                f"Dimension mismatch: vector a has {len(a)} dims, "
                f"vector b has {len(b)} dims."
            )

        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)

        dot_product = np.dot(a_np, b_np)
        norm_a = np.sqrt(np.dot(a_np, a_np))
        norm_b = np.sqrt(np.dot(b_np, b_np))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        similarity = max(-1.0, min(1.0, float(similarity)))
        return similarity