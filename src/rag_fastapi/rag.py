"""The pipeline: ingest = chunk -> embed -> store; ask = embed query -> retrieve -> generate."""

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .llm import Generator
from .store import Chunk, Hit, VectorStore


@dataclass
class Answer:
    answer: str
    hits: list[Hit]


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore,
        index_path: Path | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.embedder = embedder
        self.generator = generator
        self.store = store
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, source: str, text: str) -> int:
        """Index a document. Re-ingesting the same source replaces it."""
        self.store.remove_source(source)
        pieces = chunk_text(text, self.chunk_size, self.chunk_overlap)
        chunks = [Chunk(id=f"{source}#{i}", source=source, text=p) for i, p in enumerate(pieces)]
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self._persist()
        return len(chunks)

    def delete(self, source: str) -> int:
        removed = self.store.remove_source(source)
        self._persist()
        return removed

    def retrieve(self, question: str, k: int) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(question), k)

    def ask(self, question: str, k: int) -> Answer:
        hits = self.retrieve(question, k)
        if not hits:
            return Answer("The index is empty. Add documents first.", [])
        return Answer(self.generator.answer(question, hits), hits)

    def _persist(self) -> None:
        if self.index_path:
            self.store.save(self.index_path)
