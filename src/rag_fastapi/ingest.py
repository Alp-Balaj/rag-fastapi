"""CLI: index every .md/.txt file under a folder.

    uv run rag-ingest data/
"""

import sys
from pathlib import Path

from .config import settings
from .embeddings import FastEmbedder
from .rag import RagPipeline
from .store import VectorStore


class _NoGenerator:
    def answer(self, question, hits):
        raise RuntimeError("ingest only")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    files = sorted(p for p in root.rglob("*") if p.suffix in {".md", ".txt"})
    if not files:
        sys.exit(f"No .md or .txt files under {root}")
    pipeline = RagPipeline(
        embedder=FastEmbedder(settings.embedding_model),
        generator=_NoGenerator(),
        store=VectorStore.load(settings.index_path),
        index_path=settings.index_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    for f in files:
        n = pipeline.ingest(str(f.relative_to(root)), f.read_text(encoding="utf-8"))
        print(f"{f}: {n} chunks")
    print(f"Index now holds {len(pipeline.store)} chunks -> {settings.index_path}")
