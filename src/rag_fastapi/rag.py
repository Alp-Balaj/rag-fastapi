"""The pipeline: ingest = chunk -> embed -> store; ask = embed query -> retrieve -> generate."""

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import chunk_text
from .embeddings import Embedder
from .llm import Generator
from .store import Chunk, Hit, VectorStore

EMBED_BATCH = 256


@dataclass
class Document:
    source: str
    text: str
    # Prepended to each chunk when embedding (not stored), e.g. "Brain release (6 - Brian)",
    # so a question can match a note by its title and folder as well as its body.
    heading: str = ""


@dataclass
class Answer:
    answer: str
    hits: list[Hit]


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0  # total chunks in the index afterwards
    skipped: list[str] = field(default_factory=list)


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
        # A sync can run while searches come in; the store must not change mid-search.
        self._lock = threading.RLock()

    def ingest(self, source: str, text: str, heading: str = "") -> int:
        """Index a document. Re-ingesting the same source replaces it."""
        chunks, vectors = self._prepare([Document(source, text, heading)])
        with self._lock:
            self.store.remove_sources({source})
            self.store.add(chunks, vectors)
            self._persist()
        return len(chunks)

    def delete(self, source: str) -> int:
        with self._lock:
            removed = self.store.remove_sources({source})
            self._persist()
        return removed

    def sync(
        self,
        origin: str,
        docs: list[Document],
        force: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> SyncReport:
        """Make the index match `docs` for one origin (e.g. a notes folder).

        Only new or changed documents are embedded; documents of this origin that
        are no longer in `docs` are removed. Sources from other origins (uploads,
        other folders) are left alone. Saves once at the end.
        """
        with self._lock:
            known = self.store.fingerprints(origin)
        report = SyncReport()
        todo: list[tuple[Document, str]] = []
        for doc in docs:
            fp = self._fingerprint(doc)
            if not force and known.get(doc.source) == fp:
                report.unchanged += 1
                continue
            todo.append((doc, fp))
            if doc.source in known:
                report.updated += 1
            else:
                report.added += 1
        seen = {d.source for d in docs}
        stale = {s for s in known if s not in seen}
        report.removed = len(stale)

        # The slow part (embedding) runs without the lock, so search keeps working.
        chunks, vectors = self._prepare([d for d, _ in todo], progress)

        with self._lock:
            self.store.remove_sources({d.source for d, _ in todo} | stale)
            self.store.add(chunks, vectors)
            for doc, fp in todo:
                self.store.meta[doc.source] = {"origin": origin, "fingerprint": fp}
            if todo or stale:
                self._persist()
            report.chunks = len(self.store)
        return report

    def retrieve(self, question: str, k: int) -> list[Hit]:
        query = self.embedder.embed_query(question)
        with self._lock:
            return self.store.search(query, k)

    def ask(self, question: str, k: int) -> Answer:
        hits = self.retrieve(question, k)
        if not hits:
            return Answer("The index is empty. Add documents first.", [])
        return Answer(self.generator.answer(question, hits), hits)

    def _prepare(
        self, docs: list[Document], progress: Callable[[int, int], None] | None = None
    ) -> tuple[list[Chunk], np.ndarray]:
        chunks: list[Chunk] = []
        inputs: list[str] = []
        for doc in docs:
            for i, piece in enumerate(chunk_text(doc.text, self.chunk_size, self.chunk_overlap)):
                chunks.append(Chunk(id=f"{doc.source}#{i}", source=doc.source, text=piece))
                inputs.append(f"{doc.heading}\n\n{piece}" if doc.heading else piece)
        if not inputs:
            return [], np.zeros((0, 0), dtype=np.float32)
        parts = []
        for start in range(0, len(inputs), EMBED_BATCH):
            parts.append(self.embedder.embed(inputs[start : start + EMBED_BATCH]))
            if progress:
                progress(min(start + EMBED_BATCH, len(inputs)), len(inputs))
        return chunks, np.vstack(parts)

    def _fingerprint(self, doc: Document) -> str:
        key = f"{self.chunk_size}:{self.chunk_overlap}\n{doc.heading}\n{doc.text}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _persist(self) -> None:
        if self.index_path:
            self.store.save(self.index_path)
