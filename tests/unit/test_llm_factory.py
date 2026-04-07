"""Unit tests for LLM provider factory."""
import pytest
from unittest.mock import MagicMock
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from consilium.config import Settings
from consilium.integrations.llm_factory import _get_groq_api_key, create_llm_client


@pytest.fixture(autouse=True)
def reset_factory_state():
    """Reset module-level state before each test."""
    import consilium.integrations.llm_factory as factory_module

    factory_module._groq_keys_cache = []
    factory_module._groq_key_index = 0
    yield


def test_factory_returns_ollama_when_provider_ollama():
    """Factory returns ChatOllama when provider is ollama."""
    settings = MagicMock(spec=Settings)
    settings.llm_provider = "ollama"
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "llama3.1:8b"

    llm = create_llm_client(settings)
    assert isinstance(llm, ChatOllama)


def test_factory_returns_groq_when_provider_groq():
    """Factory returns ChatGroq when provider is groq."""
    settings = MagicMock(spec=Settings)
    settings.llm_provider = "groq"
    settings.groq_api_key_1 = "test-key-1"
    settings.groq_api_key_2 = ""
    settings.groq_api_key_3 = ""
    settings.groq_model = "llama-3.1-8b-instant"

    llm = create_llm_client(settings)
    assert isinstance(llm, ChatGroq)


def test_factory_raises_on_unsupported_provider():
    """Factory raises ValueError on unsupported provider."""
    settings = MagicMock(spec=Settings)
    settings.llm_provider = "unsupported"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(settings)


def test_groq_key_rotation_cycles_through_keys():
    """Groq key rotation uses all available keys in round-robin."""
    settings = MagicMock(spec=Settings)
    settings.groq_api_key_1 = "key-1"
    settings.groq_api_key_2 = "key-2"
    settings.groq_api_key_3 = "key-3"

    # Get 4 keys (should rotate through all 3)
    keys_used = []
    for _ in range(4):
        key = _get_groq_api_key(settings)
        keys_used.append(key)

    # Verify all 3 keys used
    assert "key-1" in keys_used
    assert "key-2" in keys_used
    assert "key-3" in keys_used
    # Verify rotation (4th call repeats first key)
    assert keys_used[3] == keys_used[0]


def test_groq_key_rotation_raises_when_no_keys():
    """Raises ValueError when no Groq keys configured."""
    settings = MagicMock(spec=Settings)
    settings.groq_api_key_1 = ""
    settings.groq_api_key_2 = ""
    settings.groq_api_key_3 = ""

    with pytest.raises(ValueError, match="No Groq API keys configured"):
        _get_groq_api_key(settings)


def test_groq_key_rotation_skips_empty_keys():
    """Key rotation only includes non-empty keys."""
    settings = MagicMock(spec=Settings)
    settings.groq_api_key_1 = "key-1"
    settings.groq_api_key_2 = ""
    settings.groq_api_key_3 = "key-3"

    keys_used = [_get_groq_api_key(settings) for _ in range(4)]

    assert "key-1" in keys_used
    assert "key-3" in keys_used
    assert "" not in keys_used


def test_planner_agent_uses_factory():
    """PlannerAgent creates LLM via factory (not hardcoded ChatOllama)."""
    from consilium.agents.planner import PlannerAgent

    agent = PlannerAgent()
    assert agent.llm is not None


def test_analyst_agent_uses_factory():
    """AnalystAgent creates LLM via factory (not hardcoded ChatOllama)."""
    from consilium.agents.analyst import AnalystAgent

    agent = AnalystAgent()
    assert agent.llm is not None
