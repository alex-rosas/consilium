"""
Unit tests for PlannerAgent retry logic (Task 7).

Verifies:
- LLM is retried up to 3 attempts on transient failure
- Falls back to single-task plan after all retries exhausted
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from tenacity import wait_none

from consilium.agents.planner import PlannerAgent, PlannerInput


def _make_valid_llm_response() -> AIMessage:
    return AIMessage(
        content='[{"task_id": "task_1", "description": "Analyze compliance risks in detail", '
        '"assigned_agent": "analyst", "dependencies": []}]'
    )


def _make_planner() -> PlannerAgent:
    agent = PlannerAgent()
    agent.llm = MagicMock()  # type: ignore[attr-defined]
    return agent


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tenacity backoff so retry tests run instantly."""
    monkeypatch.setattr("consilium.agents.planner._LLM_RETRY_WAIT", wait_none())


class TestPlannerRetry:
    async def test_retries_on_transient_failure_then_succeeds(self) -> None:
        """Fail once, succeed on second attempt — exactly 2 ainvoke calls."""
        agent = _make_planner()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[
                ConnectionError("Ollama timeout"),
                _make_valid_llm_response(),
            ]
        )

        planner_input = PlannerInput(
            task="plan",
            context={},
            user_query="What are the IFRS 15 revenue recognition requirements?",
        )
        output = await agent.execute(planner_input)

        assert agent.llm.ainvoke.call_count == 2  # type: ignore[attr-defined]
        assert len(output.task_plan) == 1
        assert output.confidence == 0.85  # LLM path, not fallback

    async def test_retries_three_times_then_uses_fallback(self) -> None:
        """Fail all 3 attempts — falls back to single-task plan, confidence=0.3."""
        agent = _make_planner()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=ConnectionError("Ollama unreachable")
        )

        planner_input = PlannerInput(
            task="plan",
            context={},
            user_query="What are the IFRS 15 revenue recognition requirements?",
        )
        output = await agent.execute(planner_input)

        assert agent.llm.ainvoke.call_count == 3  # type: ignore[attr-defined]
        assert output.confidence == 0.3
        assert len(output.task_plan) == 1
        assert "planner_error" in output.result

    async def test_bad_json_is_retried(self) -> None:
        """JSON parse failure on first attempt is retried, succeeds on second."""
        agent = _make_planner()
        agent.llm.ainvoke = AsyncMock(  # type: ignore[attr-defined]
            side_effect=[
                AIMessage(content="not json at all"),
                _make_valid_llm_response(),
            ]
        )

        planner_input = PlannerInput(
            task="plan",
            context={},
            user_query="What are the IFRS 15 revenue recognition requirements?",
        )
        output = await agent.execute(planner_input)

        assert agent.llm.ainvoke.call_count == 2  # type: ignore[attr-defined]
        assert output.confidence == 0.85
