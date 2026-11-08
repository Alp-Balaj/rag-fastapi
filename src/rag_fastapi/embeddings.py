"""Step 2 of RAG: turn text into vectors.

Anthropic has no embeddings endpoint, so this runs a small local model with
fastembed (ONNX, CPU only). The first call downloads it (~130 MB).
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # heavy import, deferred

        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
