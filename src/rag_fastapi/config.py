"""Settings, read from the environment or a .env file (prefix RAG_)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    claude_model: str = "claude-opus-5-5"
    effort: str = "medium"  # low | medium | high | xhigh | max
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150
    top_k: int = 4
    index_path: Path = Path("index.json")


settings = Settings()
