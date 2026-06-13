"""Thin wrapper around sentence-transformers.

The wrapper exists so the rest of the code only ever sees an `embed(texts)`
function. If we later swap to OpenAI embeddings or a custom model, only
this module changes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Lazy-load + cache the model.

    Loading the model is ~1s on a warm cache, ~10s cold. We don't want to
    pay that during `import` because the data-generator script imports
    config-adjacent modules but never embeds anything.
    """
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings.

    Returns an (N, D) float32 array. Normalisation is on so the resulting
    vectors play nicely with cosine-similarity-based vector stores.
    """
    if not texts:
        # Hardcode the model's dim (MiniLM = 384) rather than instantiating
        # the model just to ask. Keeps the empty-input path fast.
        return np.zeros((0, 384), dtype="float32")
    return _model().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
