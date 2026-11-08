"""FastAPI app. Run with: uv run fastapi dev src/rag_fastapi/main.py"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import settings
from .embeddings import FastEmbedder
from .llm import ClaudeGenerator, RefusalError
from .rag import RagPipeline
from .store import VectorStore

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
    score: float
    text: str


class AskOut(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/")
def hello() -> dict:
    return {"message": "Hello, World! See /docs for the API."}


@app.get("/health")
def health(pipeline: Pipeline) -> dict:
    return {"status": "ok", "chunks": len(pipeline.store)}


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


@app.post("/search")
def search(body: AskIn, pipeline: Pipeline) -> list[SourceOut]:
    """Retrieval only, no LLM call. Useful for checking what the model would see."""
    hits = pipeline.retrieve(body.question, body.k)
    return [_source_out(i, h) for i, h in enumerate(hits, start=1)]


@app.post("/ask")
def ask(body: AskIn, pipeline: Pipeline) -> AskOut:
    try:
        result = pipeline.ask(body.question, body.k)
    except RefusalError as e:
        raise HTTPException(422, str(e))
    return AskOut(
        answer=result.answer,
        sources=[_source_out(i, h) for i, h in enumerate(result.hits, start=1)],
    )


def _source_out(i, hit) -> SourceOut:
    return SourceOut(index=i, source=hit.chunk.source, score=round(hit.score, 4), text=hit.chunk.text)
