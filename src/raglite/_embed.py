"""String embedder."""

from functools import partial
from typing import Literal

import numpy as np
from litellm import embedding
from tqdm.auto import tqdm, trange

from raglite._config import RAGLiteConfig
from raglite._typing import FloatMatrix, IntVector


def _embed_sentences_with_windowing(
    sentences: list[str], *, config: RAGLiteConfig | None = None
) -> FloatMatrix:
    """Embed a document's sentences with windowing."""

    def _embed_string_batch(string_batch: list[str], *, config: RAGLiteConfig) -> FloatMatrix:
        # Embed the batch of strings.
       
        # Use LiteLLM's API to embed the batch of strings.
        response = embedding(config.embedder, string_batch)
        embeddings = np.asarray([item["embedding"] for item in response["data"]])
        # Normalise the embeddings to unit norm and cast to half precision.
        if config.embedder_normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype(np.float16)
        return embeddings

    # Window the sentences with a lookback of `config.embedder_sentence_window_size - 1` sentences.
    config = config or RAGLiteConfig()
    sentence_windows = [
        "".join(sentences[max(0, i - (config.embedder_sentence_window_size - 1)) : i + 1])
        for i in range(len(sentences))
    ]
    # Embed the sentence windows in batches.
    batch_size = 64
    batch_range = (
        partial(trange, desc="Embedding", unit="batch", dynamic_ncols=True)
        if len(sentence_windows) > batch_size
        else range
    )
    batch_embeddings = [
        _embed_string_batch(sentence_windows[i : i + batch_size], config=config)
        for i in batch_range(0, len(sentence_windows), batch_size)  # type: ignore[operator]
    ]
    sentence_embeddings = np.vstack(batch_embeddings)
    return sentence_embeddings


def embed_sentences(sentences: list[str], *, config: RAGLiteConfig | None = None) -> FloatMatrix:
    """Embed the sentences of a document as a NumPy matrix with one row per sentence."""
    sentence_embeddings = _embed_sentences_with_windowing(sentences, config=config)
    return sentence_embeddings
