"""
retriever.py
------------
Retrieval pipeline: vectorize -> guardrail check -> rank -> collapse filter -> LLM call.
Updated to work with dynamic PDF-based vector stores.
"""

from __future__ import annotations
import numpy as np
from typing import Any


class GuardrailException(Exception):
    """Raised when a query is blocked by the guardrail system."""
    pass


class Retriever:
    """
    Orchestrates the full retrieval pipeline.
    Works with both static FAQ trees and dynamic PDF-uploaded stores.
    """

    def __init__(
        self,
        tree=None,
        vector_store=None,
        guardrail_store=None,
        k: int = 5,
        theta_guard: float = 0.85,
        theta_collapse: float = 0.95,
        pick_threshold: float = 0.1
    ):
        """
        Initialize the Retriever.

        Args:
            tree: Optional RoutingTree instance for tree-based routing.
            vector_store: Optional VectorStore for direct (non-tree) retrieval.
            guardrail_store: Optional VectorStore for guardrail checking.
            k: Number of results to retrieve.
            theta_guard: Guardrail threshold.
            theta_collapse: Collapse filter threshold.
            pick_threshold: Minimum similarity to consider relevant.
        """
        self.tree = tree
        self.vector_store = vector_store
        self.guardrail_store = guardrail_store
        self.k = k
        self.theta_guard = theta_guard
        self.theta_collapse = theta_collapse
        self.pick_threshold = pick_threshold

    def retrieve_from_store(self, query: str) -> list[dict]:
        """
        Retrieve directly from the vector_store (for PDF-uploaded content).
        No tree routing needed — queries the uploaded document directly.

        Args:
            query: The user's query string.

        Returns:
            List of up to k relevant chunks that passed all filters.
        """
        if self.vector_store is None or self.vector_store.size() == 0:
            return []

        # Vectorize the query
        query_vec = self.vector_store._vectorize(query)

        # Guardrail check (if guardrail store exists and has entries)
        if self.guardrail_store and self.guardrail_store.size() > 0:
            guardrail_query_vec = self.guardrail_store._vectorize(query)
            guardrail_vectors = self.guardrail_store.get_all_vectors()

            for g_vec in guardrail_vectors:
                if len(guardrail_query_vec) == len(g_vec):
                    sim = self.guardrail_store.cosine_similarity(
                        guardrail_query_vec.tolist(), g_vec.tolist()
                    )
                    if sim > self.theta_guard:
                        raise GuardrailException(
                            f"Query blocked by guardrail. Similarity {sim:.4f} exceeds "
                            f"threshold {self.theta_guard}. "
                            f"I cannot answer questions about this topic."
                        )

        # Rank all entries
        all_scored = self.vector_store.query_all_scored(query_vec)

        # Filter by pick_threshold
        candidates = [f for f in all_scored if f['score'] >= self.pick_threshold]

        # Collapse filter
        collapse_filtered = [f for f in candidates if f['score'] <= self.theta_collapse]

        # Select top-k
        results = collapse_filtered[:self.k]

        return [
            {
                'payload': r['payload'],
                'score': r['score'],
                'text': r['text']
            }
            for r in results
        ]

    def retrieve(self, query: str, query_meta: dict[str, Any] = None) -> list[dict]:
        """
        Full retrieval — either via tree routing or direct store query.

        If tree is set and query_meta is provided, use tree routing.
        Otherwise, use direct vector_store query.
        """
        if self.tree and query_meta:
            # Tree-based routing
            leaf = self.tree.traverse(query_meta)

            query_vec = leaf.faq_store._vectorize(query)

            # Guardrail check
            guardrail_query_vec = leaf.guardrail_store._vectorize(query)
            guardrail_vectors = leaf.guardrail_store.get_all_vectors()

            for g_vec in guardrail_vectors:
                if len(guardrail_query_vec) == len(g_vec):
                    sim = leaf.guardrail_store.cosine_similarity(
                        guardrail_query_vec.tolist(), g_vec.tolist()
                    )
                    if sim > self.theta_guard:
                        raise GuardrailException(
                            f"Query blocked by guardrail. Similarity {sim:.4f} exceeds "
                            f"threshold {self.theta_guard}. "
                            f"I cannot answer questions about this topic."
                        )

            all_faqs_scored = leaf.faq_store.query_all_scored(query_vec)
            candidates = [f for f in all_faqs_scored if f['score'] >= self.pick_threshold]
            collapse_filtered = [f for f in candidates if f['score'] <= self.theta_collapse]
            results = collapse_filtered[:self.k]

            return [
                {
                    'payload': r['payload'],
                    'score': r['score'],
                    'text': r['text']
                }
                for r in results
            ]
        else:
            # Direct store query (PDF mode)
            return self.retrieve_from_store(query)

    def retrieve_with_context(self, query: str, query_meta: dict[str, Any] = None) -> dict:
        """
        Retrieve and format results as LLM context.

        Returns:
            Dict with query, results, llm_context, num_results, blocked status.
        """
        try:
            results = self.retrieve(query, query_meta)
        except GuardrailException as e:
            return {
                'query': query,
                'results': [],
                'llm_context': str(e),
                'num_results': 0,
                'blocked': True
            }

        # Format context for LLM
        context_parts = []
        for i, r in enumerate(results, 1):
            payload = r['payload']
            content = payload.get('content', payload.get('answer', r['text']))
            context_parts.append(
                f"[Result {i}] (similarity: {r['score']:.4f})\n"
                f"{content}\n"
            )

        llm_context = "\n".join(context_parts) if context_parts else "No relevant information found in the uploaded document."

        return {
            'query': query,
            'results': results,
            'llm_context': llm_context,
            'num_results': len(results),
            'blocked': False
        }