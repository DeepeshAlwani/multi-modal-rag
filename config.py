"""
config.py — Single source of truth for all configuration.

Usage in any module:
    from config import settings

    path = settings.chroma_path
    model = settings.semantic_model

All values can be overridden via environment variables or a .env file.
Pydantic validates types at startup so misconfigured deployments fail fast.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    chroma_path: str = Field("./chroma_db", description="ChromaDB persistence directory")
    repos_base_dir: str = Field("./repos", description="Root directory for cloned repos")
    database_path: str = Field("users.db", description="SQLite database file")
    test_data_path: str = Field("test_data.json", description="Evaluation test set")

    # ------------------------------------------------------------------
    # Embedding models
    # ------------------------------------------------------------------
    semantic_model: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model for semantic (natural-language) embeddings",
    )
    structural_model: str = Field(
        "microsoft/codebert-base",
        description="SentenceTransformer model for structural (source-code) embeddings",
    )

    # ------------------------------------------------------------------
    # LLM / generation
    # ------------------------------------------------------------------
    ollama_url: str = Field(
        "http://localhost:11434",
        description="Base URL for the local Ollama server",
    )
    ollama_model: str = Field(
        "llama3.2:latest",
        description="Ollama model used for answer generation",
    )

    # ------------------------------------------------------------------
    # OpenRouter (judge model for evaluation)
    # ------------------------------------------------------------------
    openrouter_api_key: str = Field("", description="OpenRouter API key")
    openrouter_url: str = Field(
        "https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter completions endpoint",
    )
    judge_model: str = Field(
        "nvidia/nemotron-3-super-120b-a12b:free",
        description="Model used as the evaluation judge",
    )
    answer_model: str = Field(
        "gemma4:e4b",
        description="Ollama model used during evaluation for answer generation",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    retrieval_top_k: int = Field(10, description="Candidates fetched from each collection before RRF")
    rerank_top_n: int = Field(4, description="Documents kept after RRF fusion for the final prompt")
    rrf_k: int = Field(60, description="RRF constant k (higher = flatter score distribution)")
    context_lines: int = Field(
        10,
        description="Lines of surrounding source code included above/below each matched function",
    )
    max_source_chars: int = Field(
        4_000,
        description="Max characters from a function's source stored in the embedding",
    )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    login_rate_limit: int = Field(10, description="Max login attempts per IP per window")
    query_rate_limit: int = Field(100, description="Max queries per user per hour")
    rate_limit_window: int = Field(3600, description="Rate-limit window in seconds")

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = Field("0.0.0.0", description="Uvicorn bind host")
    api_port: int = Field(8000, description="Uvicorn bind port")
    frontend_url: str = Field("http://localhost:8501", description="Streamlit frontend URL")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    debug_judge: bool = Field(False, description="Print raw judge responses during evaluation")
    session_ttl_days: int = Field(7, description="Session token lifetime in days")


# Module-level singleton — import this everywhere
settings = Settings()