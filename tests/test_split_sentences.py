"""Test RAGLite's sentence splitting functionality."""

from pathlib import Path

from raglite._html import document_to_html
from raglite._split_sentences import split_sentences


def test_split_sentences() -> None:
    """Test splitting a document into sentences."""
    doc_path = Path(__file__).parent / "2023_12_11_Duesseldorfer_Tabelle_-2024.pdf"
    doc = document_to_html(doc_path)
    sentences = split_sentences(doc)
    assert isinstance(sentences, list)
    assert len(sentences) == 65
