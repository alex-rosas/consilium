"""LLM provider factory for swappable LLM backends."""
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from consilium.config import Settings

# Module-level state for Groq key rotation
_groq_key_index = 0
_groq_keys_cache: List[str] = []


def _get_groq_api_key(settings: Settings) -> str:
    """Round-robin across available Groq API keys."""
    global _groq_key_index, _groq_keys_cache

    # Build key list on first call
    if not _groq_keys_cache:
        for key in [settings.groq_api_key_1, settings.groq_api_key_2, settings.groq_api_key_3]:
            if key:  # Skip empty keys
                _groq_keys_cache.append(key)

    if not _groq_keys_cache:
        raise ValueError("No Groq API keys configured")

    # Round-robin selection
    key = _groq_keys_cache[_groq_key_index % len(_groq_keys_cache)]
    _groq_key_index += 1
    return key


def create_llm_client(settings: Settings) -> BaseChatModel:
    """
    Create LLM client based on configured provider.

    Args:
        settings: Application settings with LLM provider config

    Returns:
        Instantiated LLM client ready for agent use

    Raises:
        ValueError: If provider is unsupported or config is invalid
    """
    if settings.llm_provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,  # Fixed for determinism
        )
    elif settings.llm_provider == "groq":
        return ChatGroq(
            api_key=_get_groq_api_key(settings),
            model=settings.groq_model,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
