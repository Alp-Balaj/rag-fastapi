# rag-fastapi

Day 1: RAG pipeline concepts with Python + FastAPI.

A small retrieval-augmented generation service: you add documents, it splits and embeds them locally, and `/ask` answers questions with Claude using only the retrieved passages, with `[n]` citations.

```
ingest:  text ─► chunk_text ─► FastEmbedder.embed ─► VectorStore (index.json)
ask:     question ─► embed_query ─► VectorStore.search (top k, cosine) ─► Claude ─► answer + sources
```

## Run it

```sh
cp .env.example .env            # put your ANTHROPIC_API_KEY in it
uv sync
uv run rag-ingest data/         # index the sample notes (first run downloads a ~130 MB embedding model)
uv run fastapi dev src/rag_fastapi/main.py
```

Open http://127.0.0.1:8000/docs for the interactive API, or:

```sh
curl localhost:8000/                                   # hello world
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is chunk overlap for?"}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"question": "cosine similarity", "k": 2}'    # retrieval only, no API call
curl -F file=@some-notes.md localhost:8000/documents/upload
```

Only `/ask` calls Claude; everything else runs offline.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Hello world |
| GET | `/health` | Status and chunk count |
| GET | `/documents` | Indexed sources and chunk counts |
| POST | `/documents` | `{source, text}`: index a document (same source replaces it) |
| POST | `/documents/upload` | Upload a UTF-8 `.md`/`.txt` file |
| DELETE | `/documents/{source}` | Remove a document |
| POST | `/search` | `{question, k}`: top-k chunks with scores |
| POST | `/ask` | `{question, k}`: grounded answer plus the sources used |

## Code map (read in this order)

1. `src/rag_fastapi/chunking.py`: paragraph-aware chunking with overlap
2. `src/rag_fastapi/embeddings.py`: local embeddings via fastembed (Anthropic has no embeddings API)
3. `src/rag_fastapi/store.py`: a numpy vector store, brute-force cosine similarity, JSON persistence
4. `src/rag_fastapi/llm.py`: prompt with numbered `<source>` tags, Claude call, refusal handling
5. `src/rag_fastapi/rag.py`: the pipeline gluing the four together
6. `src/rag_fastapi/main.py`: FastAPI routes, dependency injection via `Depends(get_pipeline)`

## Claude settings

- Model `claude-opus-5-5`, effort `medium` (change with `RAG_CLAUDE_MODEL` / `RAG_EFFORT`).
- Server-side refusal fallback is on (`fallbacks="default"`): if a safety classifier declines, the API retries on Anthropic's recommended fallback model. If the whole chain declines, `/ask` returns 422.

## Tests

```sh
uv run pytest
```

Tests use a bag-of-words embedder and a fake generator, so they need no model download and no API key.

## Things to try

- **See a retrieval miss.** `/search` with "how do I swap a dependency in tests?" ranks a RAG chunk above the FastAPI note; "how do I override a FastAPI dependency in tests?" gets it right. Vague wording is the most common RAG failure.
- Change `RAG_CHUNK_SIZE` to 300 and re-ingest; compare `/search` results.
- Add hybrid search: score with keyword overlap too and blend it with cosine.
- Swap `VectorStore` for Chroma or pgvector behind the same `add` / `search` interface.
