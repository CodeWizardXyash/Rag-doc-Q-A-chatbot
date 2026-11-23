"""Application configuration, loaded from environment variables / a `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Every field can be overridden by an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "RAG Document Q&A Chatbot"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:8501"]

    # --- LLM provider ---
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)

    # --- Storage ---
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "documents"
    upload_dir: Path = Path("data/uploads")

    # --- RAG tuning ---
    chunk_size: int = Field(1000, ge=100)
    chunk_overlap: int = Field(200, ge=0)
    top_k: int = Field(4, ge=1, le=20)
    max_upload_mb: int = Field(20, ge=1)

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def active_api_key(self) -> str:
        """API key for whichever provider is selected."""
        return self.gemini_api_key if self.llm_provider == "gemini" else self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (use as a FastAPI dependency)."""
    return Settings()
