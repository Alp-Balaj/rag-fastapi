"""Settings, read from the environment or a .env file (prefix RAG_)."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Settings below only reads RAG_* variables from .env. Load the whole file into
# the environment too, so the Anthropic client finds ANTHROPIC_API_KEY there.
# Variables already set in the shell win over .env.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    claude_model: str = "claude-opus-5-5"
    effort: str = "medium"  # low | medium | high | xhigh | max
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150
    top_k: int = 4
    index_path: Path = Path("index.json")  # vectors go beside it, in index.npy

    # Your notes folder (e.g. an Obsidian vault). Used by `rag-ingest` with no
    # argument and by the web UI's Re-sync button.
    notes_dir: Path | None = None
    # Comma-separated folders (or files) to skip, relative to notes_dir,
    # e.g. "7 - Dev Folder, 6 - Brian/Private". Hidden folders are always skipped.
    notes_exclude: str = ""
    # Vault name for "Open in Obsidian" links; defaults to the folder's name.
    obsidian_vault: str = ""

    @field_validator("notes_dir", mode="before")
    @classmethod
    def _expand_home(cls, v):
        return Path(v).expanduser() if v else None


settings = Settings()
