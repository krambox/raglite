"""RAGLite config."""

import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL

# Suppress rerankers output on import until [1] is fixed.
# [1] https://github.com/AnswerDotAI/rerankers/issues/36
with contextlib.redirect_stdout(StringIO()):
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.models.ranker import BaseRanker


cache_path = Path(user_data_dir("raglite", ensure_exists=True))


@dataclass(frozen=True)
class RAGLiteConfig:
    """RAGLite config."""

    # Database config.
    db_url: str | URL = f"sqlite:///{(cache_path / 'raglite.db').as_posix()}"
    # LLM config used for generation.
    llm: str = field(
        default_factory=lambda: ("gpt-4o-mini" )
    )
    llm_max_tries: int = 4
    # Embedder config used for indexing.
    embedder: str = field(
        default_factory=lambda: (  "ollama/bge-m3")
    )
    embedder_normalize: bool = True
    embedder_sentence_window_size: int = 3
    # Chunk config used to partition documents into chunks.
    chunk_max_size: int = 1440  # Max number of characters per chunk.
    # Vector search config.
    vector_search_index_metric: Literal["cosine", "dot", "l1", "l2"] = "cosine"
    vector_search_query_adapter: bool = True  # Only supported for "cosine" and "dot" metrics.
    # Reranking config.
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...] | None = field(
        default_factory=lambda: (
            ("en", FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0, cache_dir=cache_path)),
            ("other", FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0, cache_dir=cache_path)),
        ),
        compare=False,  # Exclude the reranker from comparison to avoid lru_cache misses.
    )

