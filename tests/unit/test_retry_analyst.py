"""
Unit tests for AnalystAgent retry logic (Task 7).

Verifies:
- LLM is retried up to 3 attempts on transient failure
- Falls back to rule-based classification after all retries exhausted
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from tenacity import wait_none

from consilium.agents.analyst import AnalystAgent, AnalystInput
from consilium.integrations.retrieval_mock import RetrievalResult


def _make_chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_text="The entity must recognise revenue when obligations are satisfied.",
        score=0.9,
        metadata={"document": "IFRS15", "section": "31"},
    )


def _make_valid_llm_response() -> AIMessage:
    return AIMessage(
        content='[{"clause_reference": "IFRS 15.31", "finding": "Revenue recognition '
        'obligation requires immediate attention under IFRS 15.", '
        '"risk_level": "High", "document_source": "IFRS15"}]'
    )


def _make_analyst() -> AnalystAgent:
    with patch("consilium.agents.analyst.ChatOllama"):
        agent = AnalystAgent()
    return agent


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tenacity backoff so retry tests run instantly."""
    monkeypatch.setattr("consilium.agents.analyst._LLM_RETRY_WAIT", wait_none())


class TestAnalystRetry:
    async def test_retries_on_transient_failure_then_succeeds(self) -> None:
        """Fail once, succeed on second attempt — exactly 2 ainvoke calls."""
        agent = _make_analyst()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[
                ConnectionError("Ollama timeout"),
                _make_valid_llm_response(),
            ]
        )

        analyst_input = AnalystInput(
            task="classify",
            context={},
            retrieved_chunks=[_make_chunk()],
        )
        output = await agent.execute(analyst_input)

        assert agent.llm.ainvoke.call_count == 2  # type: ignore[attr-defined]
        assert len(output.risk_findings) == 1
        assert output.confidence == 0.85  # LLM path, not fallback

    async def test_retries_three_times_then_uses_fallback(self) -> None:
        """Fail all 3 attempts — falls back to rule-based, confidence=0.3."""
        agent = _make_analyst()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=ConnectionError("Ollama unreachable")
        )

        analyst_input = AnalystInput(
            task="classify",
            context={},
            retrieved_chunks=[_make_chunk()],
        )
        output = await agent.execute(analyst_input)

        assert agent.llm.ainvoke.call_count == 3  # type: ignore[attr-defined]
        assert output.confidence == 0.3
        assert "analyst_error" in output.result

    async def test_bad_json_is_retried(self) -> None:
        """JSON parse failure on first attempt is retried, succeeds on second."""
        agent = _make_analyst()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[
                AIMessage(content="this is not json"),
                _make_valid_llm_response(),
            ]
        )

        analyst_input = AnalystInput(
            task="classify",
            context={},
            retrieved_chunks=[_make_chunk()],
        )
        output = await agent.execute(analyst_input)

        assert agent.llm.ainvoke.call_count == 2  # type: ignore[attr-defined]
        assert output.confidence == 0.85
