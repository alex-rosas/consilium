"""
Configuration management for Consilium.
All settings loaded from environment variables.
Never hardcode values — everything goes through Settings.
"""

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
    llm_provider: Literal["ollama", "groq", "together"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # Logging
    log_level: str = "INFO"


# Global settings instance — import from here everywhere
settings = Settings()
