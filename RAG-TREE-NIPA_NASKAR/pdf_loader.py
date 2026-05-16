"""
pdf_loader.py
-------------
Extracts text from uploaded PDF files and splits into chunks
for ingestion into the VectorStore.
"""

from __future__ import annotations
import fitz  # PyMuPDF
from typing import Any


class PDFLoader:
    """
    Loads PDF files, extracts text, and splits into chunks
    suitable for vector store ingestion.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize the PDF loader.

        Args:
            chunk_size: Maximum number of words per chunk.
            chunk_overlap: Number of overlapping words between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Full text content of the PDF as a single string.
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract all text from PDF bytes (for Streamlit file uploader).

        Args:
            pdf_bytes: Raw bytes of the PDF file.

        Returns:
            Full text content of the PDF as a single string.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks based on word count.

        Strategy:
        - Split text into words
        - Create chunks of chunk_size words
        - Each chunk overlaps with the next by chunk_overlap words
        - This ensures no information is lost at chunk boundaries

        Args:
            text: The full text to split.

        Returns:
            List of text chunks.
        """
        # Clean the text
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())  # Normalize whitespace

        words = text.split()

        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            # Move start forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def load_pdf(self, pdf_path: str) -> list[dict]:
        """
        Full pipeline: extract text from PDF file path → chunk → return as list of dicts.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dicts with 'text' and 'payload' keys, ready for VectorStore.add_bulk().
        """
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)

        items = []
        for i, chunk in enumerate(chunks):
            items.append({
                'text': chunk,
                'payload': {
                    'source': pdf_path,
                    'chunk_id': i,
                    'content': chunk
                }
            })

        return items

    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> list[dict]:
        """
        Full pipeline: extract text from PDF bytes → chunk → return as list of dicts.

        Args:
            pdf_bytes: Raw bytes of the PDF file.
            filename: Original filename for metadata.

        Returns:
            List of dicts with 'text' and 'payload' keys, ready for VectorStore.add_bulk().
        """
        text = self.extract_text_from_bytes(pdf_bytes)
        chunks = self.chunk_text(text)

        items = []
        for i, chunk in enumerate(chunks):
            items.append({
                'text': chunk,
                'payload': {
                    'source': filename,
                    'chunk_id': i,
                    'content': chunk
                }
            })

        return items