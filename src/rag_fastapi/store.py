"""Step 3 of RAG: a vector store.

Deliberately tiny: a numpy matrix plus metadata, cosine similarity by brute
force. Chunks and per-source metadata go to a JSON file, vectors to a .npy file
beside it (index.json + index.npy). Fine up to tens of thousands of chunks; past
that you would swap in Chroma, pgvector, Qdrant, etc. The interface stays the same.
"""

import json
import os
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
        # Per source: where it came from and a fingerprint of what was embedded,
        # so a sync can skip unchanged documents. Sources added by hand have none.
        self.meta: dict[str, dict[str, str]] = {}

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

    def remove_sources(self, sources: set[str]) -> int:
        """Remove every chunk of these sources; returns how many chunks went."""
        for s in sources:
            self.meta.pop(s, None)
        keep = [i for i, c in enumerate(self.chunks) if c.source not in sources]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        return removed

    def fingerprints(self, origin: str) -> dict[str, str]:
        return {s: m["fingerprint"] for s, m in self.meta.items() if m.get("origin") == origin}

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
        # Write to temp files and rename, so a crash mid-save never leaves a broken index.
        vectors_path = _vectors_path(path)
        tmp_vectors = vectors_path.with_name(vectors_path.name + ".tmp")
        with open(tmp_vectors, "wb") as f:
            np.save(f, self.vectors)
        tmp_json = path.with_name(path.name + ".tmp")
        data = {"chunks": [asdict(c) for c in self.chunks], "meta": self.meta}
        tmp_json.write_text(json.dumps(data, ensure_ascii=False))
        os.replace(tmp_vectors, vectors_path)
        os.replace(tmp_json, path)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        if not path.exists():
            return store
        data = json.loads(path.read_text())
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        store.meta = data.get("meta", {})
        if "vectors" in data:  # older format: vectors inline in the JSON
            store.vectors = np.array(data["vectors"], dtype=np.float32)
        elif _vectors_path(path).exists():
            store.vectors = np.load(_vectors_path(path))
        if len(store.vectors) != len(store.chunks):
            raise ValueError(f"{path} and its vectors disagree; rebuild the index")
        return store


def _vectors_path(path: Path) -> Path:
    return path.with_suffix(".npy")


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
