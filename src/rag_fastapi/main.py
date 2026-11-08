"""FastAPI app. Run with: uv run fastapi dev src/rag_fastapi/main.py"""

import threading
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from .embeddings import FastEmbedder
from .llm import ClaudeGenerator, GenerationError, RefusalError
from .rag import RagPipeline
from .store import Hit, VectorStore
from .vault import parse_exclude, read_notes

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="RAG over your documents", version="0.1.0")


@lru_cache
def get_pipeline() -> RagPipeline:
    return RagPipeline(
        embedder=FastEmbedder(settings.embedding_model),
        generator=ClaudeGenerator(settings.claude_model, settings.effort),
        store=VectorStore.load(settings.index_path),
        index_path=settings.index_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]


class DocumentIn(BaseModel):
    source: str = Field(examples=["notes/rag.md"])
    text: str


class AskIn(BaseModel):
    question: str
    k: int = Field(default=settings.top_k, ge=1, le=20)


class SourceOut(BaseModel):
    index: int
    source: str
    title: str
    folder: str
    score: float
    text: str
    link: str | None = None  # obsidian:// link, for notes from the notes folder


class AskOut(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(pipeline: Pipeline) -> dict:
    return {
        "status": "ok",
        "chunks": len(pipeline.store),
        "documents": len(pipeline.store.sources()),
        "notes_dir": str(settings.notes_dir) if settings.notes_dir else None,
    }


@app.get("/documents")
def list_documents(pipeline: Pipeline) -> dict[str, int]:
    """Indexed sources and how many chunks each has."""
    return pipeline.store.sources()


@app.post("/documents", status_code=201)
def add_document(doc: DocumentIn, pipeline: Pipeline) -> dict:
    return {"source": doc.source, "chunks": pipeline.ingest(doc.source, doc.text)}


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile, pipeline: Pipeline) -> dict:
    """Upload a .txt or .md file."""
    try:
        text = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, "Only UTF-8 text files are supported")
    source = file.filename or "upload"
    return {"source": source, "chunks": pipeline.ingest(source, text)}


@app.delete("/documents/{source:path}")
def delete_document(source: str, pipeline: Pipeline) -> dict:
    removed = pipeline.delete(source)
    if not removed:
        raise HTTPException(404, f"No document named {source!r}")
    return {"source": source, "removed_chunks": removed}


_sync_lock = threading.Lock()


@app.post("/sync")
def sync_notes(pipeline: Pipeline) -> dict:
    """Bring the index up to date with the notes folder (RAG_NOTES_DIR).

    Only new or changed notes are embedded, so after the first run this takes seconds.
    """
    root = _notes_root()
    if root is None:
        raise HTTPException(400, "Set RAG_NOTES_DIR in .env to your notes folder, then restart.")
    if not root.is_dir():
        raise HTTPException(400, f"RAG_NOTES_DIR is not a folder: {root}")
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(409, "A sync is already running.")
    try:
        docs, skipped = read_notes(root, parse_exclude(settings.notes_exclude))
        report = pipeline.sync(str(root), docs)
    finally:
        _sync_lock.release()
    report.skipped = skipped
    return vars(report)


@app.post("/search")
def search(body: AskIn, pipeline: Pipeline) -> list[SourceOut]:
    """Retrieval only, no LLM call. Useful for checking what the model would see."""
    hits = pipeline.retrieve(body.question, body.k)
    return [_source_out(i, h, pipeline) for i, h in enumerate(hits, start=1)]


@app.post("/ask")
def ask(body: AskIn, pipeline: Pipeline) -> AskOut:
    try:
        result = pipeline.ask(body.question, body.k)
    except RefusalError as e:
        raise HTTPException(422, str(e))
    except GenerationError as e:
        raise HTTPException(502, str(e))
    return AskOut(
        answer=result.answer,
        sources=[_source_out(i, h, pipeline) for i, h in enumerate(result.hits, start=1)],
    )


def _source_out(i: int, hit: Hit, pipeline: RagPipeline) -> SourceOut:
    source = hit.chunk.source
    path = PurePosixPath(source)
    root = _notes_root()
    from_notes = root is not None and pipeline.store.meta.get(source, {}).get("origin") == str(root)
    return SourceOut(
        index=i,
        source=source,
        title=path.stem,
        folder="" if str(path.parent) == "." else str(path.parent),
        score=round(hit.score, 4),
        text=hit.chunk.text,
        link=_obsidian_link(root, source) if from_notes else None,
    )


def _notes_root() -> Path | None:
    return settings.notes_dir.resolve() if settings.notes_dir else None


def _obsidian_link(root: Path, source: str) -> str:
    vault = settings.obsidian_vault or root.name
    file = source[:-3] if source.endswith(".md") else source
    return f"obsidian://open?vault={quote(vault, safe='')}&file={quote(file, safe='')}"
