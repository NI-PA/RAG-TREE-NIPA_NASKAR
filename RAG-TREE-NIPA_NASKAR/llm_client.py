"""
llm_client.py
-------------
Thin wrapper for LLM API (Claude or OpenAI).
API key is loaded from environment variable.
"""

from __future__ import annotations
import os
from typing import Optional


class LLMClient:
    """
    A thin wrapper around an LLM API (supports OpenAI and Anthropic Claude).
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            provider: Either "openai" or "anthropic".
            model: Model name.
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
            self.model = model or "gpt-4"
            self.api_url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or "claude-3-sonnet-20240229"
            self.api_url = "https://api.anthropic.com/v1/messages"
        else:
            raise ValueError(f"Unsupported provider: {provider}.")

        if not self.api_key:
            pass  # Silent — Streamlit handles the display

    def generate(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM given a query and retrieved context."""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the context does not contain enough information, say so clearly. "
                "Do not make up information."
            )

        user_message = (
            f"Context (retrieved from document):\n"
            f"{context}\n\n"
            f"User Question: {query}\n\n"
            f"Please answer the question based on the context above."
        )

        if not self.api_key:
            return self._simulate_response(query, context)

        try:
            import requests

            if self.provider == "openai":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024
                }
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            elif self.provider == "anthropic":
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_message}
                    ]
                }
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()["content"][0]["text"]

        except Exception as e:
            return f"[LLM API Error: {e}]\n" + self._simulate_response(query, context)

    def _simulate_response(self, query: str, context: str) -> str:
        """Simulate an LLM response for demo purposes."""
        if "No relevant" in context:
            return (
                "I couldn't find relevant information in the uploaded document "
                "for your question. Try rephrasing or upload a different document."
            )
        # Extract meaningful content from context
        lines = context.split("\n")
        content_lines = [l for l in lines if l.strip() and not l.startswith("[Result")]
        summary = " ".join(content_lines[:3])[:500]
        
        return (
            f"Based on the uploaded document, here is the relevant information:\n\n"
            f"{summary}\n\n"
            f"[Simulated response - set OPENAI_API_KEY or ANTHROPIC_API_KEY for real LLM answers]"
        )