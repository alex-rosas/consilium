"""
Configuration management for Consilium.
All settings loaded from environment variables.
Never hardcode values — everything goes through Settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    llm_provider: Literal["ollama", "groq"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Groq
    groq_api_key_1: str = Field(default="", description="Primary Groq API key")
    groq_api_key_2: str = Field(default="", description="Secondary Groq API key")
    groq_api_key_3: str = Field(default="", description="Tertiary Groq API key")
    groq_model: str = Field(default="llama-3.1-8b-instant")

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # Quaestor Integration (Phase 2+)
    quaestor_base_url: str = "http://localhost:8000"
    # Controls which retrieval backend is used:
    #   "mock"     → MockRetrieval (deterministic, no external dependency)
    #   "quaestor" → QuaestorClient (real HTTP calls to Quaestor service)
    retrieval_provider: Literal["mock", "quaestor"] = "mock"
    # When True: QuaestorClient connection errors fall back to MockRetrieval (dev mode)
    # When False (default): connection errors raise HTTP 503 (production correctness)
    allow_mock_fallback: bool = False

    # Logging
    log_level: str = "INFO"


# Global settings instance — import from here everywhere
settings = Settings()
