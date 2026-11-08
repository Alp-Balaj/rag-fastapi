"""Offline tests: a fake embedder and a fake LLM, so no model download or API key."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_fastapi.chunking import chunk_text
from rag_fastapi.config import settings
from rag_fastapi.llm import build_prompt
from rag_fastapi.main import app, get_pipeline
from rag_fastapi.rag import Document, RagPipeline
from rag_fastapi.store import Chunk, Hit, VectorStore
from rag_fastapi.vault import clean_markdown, find_note_files, parse_exclude, read_notes

VOCAB = ["fastapi", "python", "rag", "retrieval", "langgraph", "graph", "crewai", "agents"]


class BagOfWordsEmbedder:
    """Counts vocabulary words. Crude, but similar texts get similar vectors."""

    def embed(self, texts):
        return np.array([self.embed_query(t) for t in texts])

    def embed_query(self, text):
        words = text.lower().split()
        return np.array([sum(w.strip(".,?") == v for w in words) for v in VOCAB], dtype=np.float32) + 1e-3


class CountingEmbedder(BagOfWordsEmbedder):
    def __init__(self):
        self.embedded = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


class EchoGenerator:
    def __init__(self):
        self.calls = []

    def answer(self, question, hits):
        self.calls.append((question, hits))
        return f"answer from {hits[0].chunk.source} [1]"


@pytest.fixture
def client(tmp_path):
    generator = EchoGenerator()
    pipeline = RagPipeline(BagOfWordsEmbedder(), generator, VectorStore(), tmp_path / "index.json")
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as c:
        c.pipeline, c.generator = pipeline, generator
        yield c
    app.dependency_overrides.clear()


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # consecutive chunks share some text
    assert chunks[0][-40:].split()[-1] in chunks[1]


def test_chunk_splits_one_huge_paragraph():
    chunks = chunk_text("x" * 2000, size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) >= 4


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_store_ranks_by_cosine_similarity():
    store = VectorStore()
    store.add(
        [Chunk("a", "a", "a"), Chunk("b", "b", "b")],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    hits = store.search(np.array([0.9, 0.1]), k=2)
    assert [h.chunk.id for h in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


def test_store_round_trips_to_disk(tmp_path):
    store = VectorStore()
    store.add([Chunk("a", "s", "hello")], np.array([[3.0, 4.0]]))
    store.meta["s"] = {"origin": "/notes", "fingerprint": "abc"}
    store.save(tmp_path / "i.json")
    assert (tmp_path / "i.npy").exists()
    loaded = VectorStore.load(tmp_path / "i.json")
    assert loaded.chunks == store.chunks
    assert loaded.meta == store.meta
    np.testing.assert_allclose(loaded.vectors, [[0.6, 0.8]])


def test_store_loads_old_inline_json_format(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"chunks": [{"id": "a#0", "source": "a", "text": "t"}], "vectors": [[1.0, 0.0]]}')
    loaded = VectorStore.load(path)
    assert [c.source for c in loaded.chunks] == ["a"] and loaded.meta == {}


def test_prompt_numbers_sources():
    hits = [Hit(Chunk("1", "doc.md", "alpha"), 0.9), Hit(Chunk("2", "b.md", "beta"), 0.5)]
    prompt = build_prompt("q?", hits)
    assert '<source index="1" name="doc.md">' in prompt
    assert '<source index="2" name="b.md">' in prompt
    assert prompt.endswith("Question: q?")


def test_web_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Ask your" in r.text


def test_ingest_search_ask_delete(client):
    client.post("/documents", json={"source": "web.md", "text": "FastAPI is a Python web framework."})
    client.post("/documents", json={"source": "graph.md", "text": "LangGraph builds agents as a graph."})
    assert client.get("/documents").json() == {"web.md": 1, "graph.md": 1}

    hits = client.post("/search", json={"question": "which python framework?", "k": 1}).json()
    assert hits[0]["source"] == "web.md"

    body = client.post("/ask", json={"question": "graph agents?", "k": 2}).json()
    assert body["answer"] == "answer from graph.md [1]"
    assert [s["index"] for s in body["sources"]] == [1, 2]

    assert client.delete("/documents/graph.md").json()["removed_chunks"] == 1
    assert client.delete("/documents/graph.md").status_code == 404
    assert client.get("/health").json()["chunks"] == 1


def test_reingest_replaces_document(client):
    client.post("/documents", json={"source": "a.md", "text": "old"})
    client.post("/documents", json={"source": "a.md", "text": "new"})
    assert [c.text for c in client.pipeline.store.chunks] == ["new"]


def test_ask_on_empty_index_skips_llm(client):
    body = client.post("/ask", json={"question": "anything?"}).json()
    assert "empty" in body["answer"]
    assert client.generator.calls == []


def test_upload(client):
    r = client.post("/documents/upload", files={"file": ("notes.md", b"RAG means retrieval augmented generation.")})
    assert r.status_code == 201 and r.json() == {"source": "notes.md", "chunks": 1}


# --- notes folder ---------------------------------------------------------------


def test_clean_markdown_strips_obsidian_syntax():
    raw = "---\ntags: [x]\n---\nSee [[Other note]] and [[folder/Deep|the deep one]].\n![[pic.png]]%% hidden %%\n[[Note#Heading]]"
    assert clean_markdown(raw) == "See Other note and the deep one.\n\nNote"


def test_find_note_files_skips_hidden_and_excluded(tmp_path):
    for rel in ["a.md", "keep/b.txt", "keep/c.pdf", ".obsidian/x.md", "private/d.md", "keep/secret.md", ".hidden.md"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("hi")
    found = [p.relative_to(tmp_path).as_posix() for p in find_note_files(tmp_path, parse_exclude("private/, keep/secret.md"))]
    assert found == ["a.md", "keep/b.txt"]


def test_read_notes_adds_heading_and_skips_unreadable(tmp_path):
    (tmp_path / "Folder").mkdir()
    (tmp_path / "Folder" / "My note.md").write_text("body")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad")
    docs, skipped = read_notes(tmp_path)
    assert docs == [Document("Folder/My note.md", "body", "My note (Folder)")]
    assert skipped == ["bad.md: UnicodeDecodeError"]


def test_sync_only_embeds_changes_and_drops_deleted(tmp_path):
    embedder = CountingEmbedder()
    pipeline = RagPipeline(embedder, EchoGenerator(), VectorStore(), tmp_path / "index.json")
    pipeline.ingest("upload.md", "an uploaded python doc")  # another origin: must survive syncs
    docs = [Document("a.md", "rag retrieval", "a"), Document("b.md", "fastapi python", "b")]

    first = pipeline.sync("/notes", docs)
    assert (first.added, first.updated, first.removed, first.unchanged) == (2, 0, 0, 0)
    assert "a\n\nrag retrieval" in embedder.embedded  # heading is embedded with the chunk

    embedder.embedded.clear()
    second = pipeline.sync("/notes", [Document("a.md", "rag retrieval graph", "a")])
    assert (second.added, second.updated, second.removed, second.unchanged) == (0, 1, 1, 0)
    assert embedder.embedded == ["a\n\nrag retrieval graph"]
    assert set(pipeline.store.sources()) == {"upload.md", "a.md"}

    embedder.embedded.clear()
    third = pipeline.sync("/notes", [Document("a.md", "rag retrieval graph", "a")])
    assert (third.unchanged, embedder.embedded) == (1, [])

    reloaded = VectorStore.load(tmp_path / "index.json")
    assert set(reloaded.sources()) == {"upload.md", "a.md"}
    assert reloaded.fingerprints("/notes").keys() == {"a.md"}


def test_sync_endpoint_and_obsidian_links(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "notes_dir", None)
    assert client.post("/sync").status_code == 400  # no notes folder configured

    vault = tmp_path / "My Vault"
    (vault / "Dev").mkdir(parents=True)
    (vault / "Dev" / "Py notes.md").write_text("python fastapi tips")
    monkeypatch.setattr(settings, "notes_dir", vault)

    report = client.post("/sync").json()
    assert (report["added"], report["chunks"]) == (1, 1)

    hit = client.post("/search", json={"question": "python", "k": 1}).json()[0]
    assert (hit["title"], hit["folder"]) == ("Py notes", "Dev")
    assert hit["link"] == "obsidian://open?vault=My%20Vault&file=Dev%2FPy%20notes"
    assert client.get("/health").json()["notes_dir"] == str(vault)
