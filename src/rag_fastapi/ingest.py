"""CLI: sync every .md/.txt file under a folder into the index.

    uv run rag-ingest                    # your notes folder (RAG_NOTES_DIR)
    uv run rag-ingest data/              # any other folder
    uv run rag-ingest --rebuild          # re-embed everything, e.g. after changing the embedding model

Only new or changed files are embedded; files deleted from the folder are dropped
from the index. Stop the server first (or use the web UI's Re-sync button instead):
the running server keeps its own copy of the index and would overwrite this one.
"""

import argparse
import sys
import time
from pathlib import Path

from .config import settings
from .embeddings import FastEmbedder
from .rag import RagPipeline
from .store import VectorStore
from .vault import parse_exclude, read_notes


class _NoGenerator:
    def answer(self, question, hits):
        raise RuntimeError("ingest only")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-ingest", description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", nargs="?", type=Path, help="defaults to RAG_NOTES_DIR, else data/")
    parser.add_argument("--rebuild", action="store_true", help="re-embed every file")
    args = parser.parse_args()

    root = (args.folder or settings.notes_dir or Path("data")).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    exclude = parse_exclude(settings.notes_exclude) if root == _resolved_notes_dir() else set()

    docs, skipped = read_notes(root, exclude)
    if not docs:
        sys.exit(f"No .md or .txt files under {root}")
    print(f"{len(docs)} files under {root}" + (f" (excluding {', '.join(sorted(exclude))})" if exclude else ""))

    pipeline = RagPipeline(
        embedder=FastEmbedder(settings.embedding_model),
        generator=_NoGenerator(),
        store=VectorStore.load(settings.index_path),
        index_path=settings.index_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        print(f"\r  embedding {done}/{total} chunks", end="", flush=True)

    report = pipeline.sync(str(root), docs, force=args.rebuild, progress=progress)
    if report.added or report.updated:
        print()
    print(
        f"added {report.added}, updated {report.updated}, removed {report.removed}, "
        f"unchanged {report.unchanged} in {time.monotonic() - started:.1f}s"
    )
    for s in skipped:
        print(f"  skipped {s}")
    print(f"Index now holds {report.chunks} chunks -> {settings.index_path}")


def _resolved_notes_dir() -> Path | None:
    return settings.notes_dir.resolve() if settings.notes_dir else None
