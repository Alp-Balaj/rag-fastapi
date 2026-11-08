"""Step 3 of RAG: a vector store.

Deliberately tiny: a numpy matrix plus metadata, cosine similarity by brute
force, saved to JSON. Fine up to tens of thousands of chunks; past that you
would swap in Chroma, pgvector, Qdrant, etc. The interface stays the same.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    source: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk")
        if not chunks:
            return
        vectors = _normalize(vectors.astype(np.float32))
        self.vectors = vectors if len(self) == 0 else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def remove_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        return removed

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        if len(self) == 0:
            return []
        q = _normalize(query_vector.astype(np.float32).reshape(1, -1))[0]
        scores = self.vectors @ q  # cosine similarity, since both sides are unit length
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def save(self, path: Path) -> None:
        data = {"chunks": [asdict(c) for c in self.chunks], "vectors": self.vectors.tolist()}
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        if path.exists():
            data = json.loads(path.read_text())
            store.chunks = [Chunk(**c) for c in data["chunks"]]
            store.vectors = np.array(data["vectors"], dtype=np.float32)
        return store


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
