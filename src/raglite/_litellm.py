import asyncio
import contextlib
import logging
import os
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from functools import cache
from io import StringIO
from typing import Any, ClassVar, cast

import httpx
import litellm
from litellm import (  # type: ignore[attr-defined]
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    CustomLLM,
    GenericStreamingChunk,
    ModelResponse,
    convert_to_model_response_object,
    get_model_info,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import custom_llm_setup

from raglite._config import RAGLiteConfig

# Reduce the logging level for LiteLLM, flashrank, and httpx.
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "WARNING"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("flashrank").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)



@cache
def get_context_size(config: RAGLiteConfig, *, fallback: int = 2048) -> int:
    """Get the context size for the configured LLM."""
    # Attempt to read the context size from LiteLLM's model info.
    model_info = get_model_info(config.llm)
    max_tokens = model_info.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens
    # Fall back to a default context size if the model info is not available.
    if fallback > 0:
        warnings.warn(
            f"Could not determine the context size of {config.llm} from LiteLLM's model_info, using {fallback}.",
            stacklevel=2,
        )
        return 2048
    error_message = f"Could not determine the context size of {config.llm}."
    raise ValueError(error_message)


@cache
def get_embedding_dim(config: RAGLiteConfig, *, fallback: bool = True) -> int:
    """Get the embedding dimension for the configured embedder."""
    # Attempt to read the embedding dimension from LiteLLM's model info.
    model_info = get_model_info(config.embedder)
    embedding_dim = model_info.get("output_vector_size")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        return embedding_dim
    # If that fails, fall back to embedding a single sentence and reading its embedding dimension.
    if fallback:
        from raglite._embed import embed_sentences

        warnings.warn(
            f"Could not determine the embedding dimension of {config.embedder} from LiteLLM's model_info, using fallback.",
            stacklevel=2,
        )
        fallback_embeddings = embed_sentences(["Hello world"], config=config)
        return fallback_embeddings.shape[1]
    error_message = f"Could not determine the embedding dimension of {config.embedder}."
    raise ValueError(error_message)
