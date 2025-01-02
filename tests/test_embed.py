"""Test RAGLite's embedding functionality."""

from pathlib import Path

import numpy as np

from raglite import RAGLiteConfig
from raglite._embed import embed_sentences
from raglite._html import document_to_html
from raglite._split_sentences import split_sentences


def test_embed(embedder: str) -> None:
    """Test embedding a document."""
    raglite_test_config = RAGLiteConfig(embedder=embedder, embedder_normalize=True)
    doc_path = Path(__file__).parent / "2023_12_11_Duesseldorfer_Tabelle_-2024.pdf" 
    doc = document_to_html(doc_path)
    sentences = split_sentences(doc, max_len=raglite_test_config.chunk_max_size)
    sentence_embeddings = embed_sentences(sentences, config=raglite_test_config)
    assert isinstance(sentences, list)
    assert isinstance(sentence_embeddings, np.ndarray)
    assert len(sentences) == len(sentence_embeddings)
    assert sentence_embeddings.shape[1] >= 128  # noqa: PLR2004
    assert sentence_embeddings.dtype == np.float16
    assert np.all(np.isfinite(sentence_embeddings))
    assert np.allclose(np.linalg.norm(sentence_embeddings, axis=1), 1.0, rtol=1e-3)
