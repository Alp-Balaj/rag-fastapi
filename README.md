# rag-fastapi

Day 1: RAG pipeline concepts with Python + FastAPI.

A small retrieval-augmented generation service: point it at your notes (an Obsidian vault or any Markdown folder), it splits and embeds them locally, and you ask questions in a web page. Claude answers using only the retrieved passages, with `[n]` citations that link to the notes.

```
ingest:  notes ─► clean_markdown ─► chunk_text ─► FastEmbedder.embed ─► VectorStore (index.json + index.npy)
ask:     question ─► embed_query ─► VectorStore.search (top k, cosine) ─► Claude ─► answer + sources
```

## Run it

```sh
cp .env.example .env            # set ANTHROPIC_API_KEY and RAG_NOTES_DIR (see Configuration)
uv sync
uv run rag-ingest               # index your notes (first run downloads a ~130 MB embedding model)
uv run fastapi dev src/rag_fastapi/main.py
```

Open http://127.0.0.1:8000 for the web UI. No notes folder yet? `uv run rag-ingest data/` indexes the two sample notes instead.

## Web UI

One input box. Enter runs it; Shift+Enter adds a line; `/` focuses the box from anywhere.

- **Ask Claude** sends the matching passages to Claude and shows the answer. Each `[n]` citation jumps to its source.
- **Search only** shows the closest passages without calling Claude: offline, free and instant. Use it to see what Claude would be given, or for notes you'd rather not send anywhere.
- Source cards show the note's title, folder, similarity score and the passage, with your question's words highlighted, and an **Open in Obsidian** link.
- **Re-sync** (top right) re-indexes notes you've added, changed or deleted since the last sync. Only changed notes are embedded, so it takes seconds.

## API

Open http://127.0.0.1:8000/docs for the interactive API, or:

```sh
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is chunk overlap for?"}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"question": "cosine similarity", "k": 2}'    # retrieval only, no API call
curl -F file=@some-notes.md localhost:8000/documents/upload
```

Only `/ask` calls Claude; everything else runs offline.

## Configuration (.env)

The app reads `.env` from the directory you start it in, so run commands from the repo root.

1. Create a key at https://console.anthropic.com under **Settings → API Keys** (the account needs credits for `/ask` to work).
2. `cp .env.example .env` and set the key on the first line, with no quotes or spaces:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Set your notes folder, and optionally folders to leave out (comma-separated, relative to the notes folder). Quote paths with spaces:
   ```
   RAG_NOTES_DIR="~/Documents/My Vault"
   RAG_NOTES_EXCLUDE="Private, Archive/Old"
   ```
   Hidden folders (`.obsidian`, `.git`, ...) are always skipped. Excluding a folder later and pressing Re-sync removes its notes from the index.
4. Restart the server after editing `.env`; it is read once at startup.

The other settings in `.env.example` (`RAG_CLAUDE_MODEL`, `RAG_TOP_K`, `RAG_CHUNK_SIZE`, ...) are optional and show their defaults.

Variables already set in your shell take precedence over `.env`. If `/ask` fails with *"Could not resolve authentication method"* even though `.env` has the key, an empty `ANTHROPIC_API_KEY` is probably set in that terminal: run `unset ANTHROPIC_API_KEY` (or open a new terminal) and restart the server.

`.env` is gitignored; never commit it.

## Notes folder

`rag-ingest` (and the Re-sync button) reads every `.md`/`.txt` file under `RAG_NOTES_DIR`:

- Frontmatter, `%% comments %%` and `![[embeds]]` are dropped; `[[links|aliases]]` become plain text.
- Each chunk is embedded together with its note's title and folder, so a question can match a note by its title, not only its body.
- The index remembers a fingerprint per note. A sync embeds only new or changed notes and removes notes that were deleted or excluded. Notes added through the API or upload are left alone.
- Files that aren't valid UTF-8 are skipped and reported, not fatal.
- `uv run rag-ingest --rebuild` re-embeds everything; do that after changing `RAG_EMBEDDING_MODEL`. Changing the chunk size triggers a re-embed by itself.

Run `rag-ingest` while the server is stopped, or use Re-sync in the UI instead: the server keeps the index in memory and would overwrite what the CLI wrote.

**Privacy:** embedding and Search only run on your machine. Ask Claude sends the question and the top passages to the Anthropic API.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | The web UI |
| GET | `/health` | Status, note and chunk counts, notes folder |
| POST | `/sync` | Re-index changed notes in `RAG_NOTES_DIR` |
| GET | `/documents` | Indexed sources and chunk counts |
| POST | `/documents` | `{source, text}`: index a document (same source replaces it) |
| POST | `/documents/upload` | Upload a UTF-8 `.md`/`.txt` file |
| DELETE | `/documents/{source}` | Remove a document |
| POST | `/search` | `{question, k}`: top-k chunks with title, folder, score, Obsidian link |
| POST | `/ask` | `{question, k}`: grounded answer plus the sources used |

## Code map (read in this order)

1. `src/rag_fastapi/chunking.py`: paragraph-aware chunking with overlap
2. `src/rag_fastapi/embeddings.py`: local embeddings via fastembed (Anthropic has no embeddings API)
3. `src/rag_fastapi/store.py`: a numpy vector store, brute-force cosine similarity, JSON + `.npy` persistence
4. `src/rag_fastapi/llm.py`: prompt with numbered `<source>` tags, Claude call, refusal and error handling
5. `src/rag_fastapi/rag.py`: the pipeline gluing the four together, including incremental `sync`
6. `src/rag_fastapi/vault.py`: reading a notes folder: skipping hidden/excluded folders, cleaning Obsidian syntax
7. `src/rag_fastapi/main.py`: FastAPI routes, dependency injection via `Depends(get_pipeline)`
8. `src/rag_fastapi/static/index.html`: the web UI (plain HTML/JS, no build step)

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
