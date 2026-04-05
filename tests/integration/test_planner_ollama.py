"""
Integration test: PlannerAgent with real ChatOllama.

Requires Ollama running at localhost:11434 with llama3.1:8b loaded.
Skip gracefully if Ollama is unavailable.

Gated with @pytest.mark.integration — not run in standard unit test suite.
"""

from __future__ import annotations

import pytest

from consilium.agents.planner import PlannerAgent, PlannerInput


def _make_input() -> PlannerInput:
    return PlannerInput(
        task="Decompose compliance query",
        context={},
        user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
    )


async def _ollama_is_available() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
async def test_planner_chatollama_returns_parseable_json() -> None:
    """ChatOllama path returns parseable JSON — no fallback triggered.

    Verification criteria:
    - output.confidence > 0.5 (fallback sets 0.3)
    - output.task_plan has >= 1 task
    - 'planner_error' is NOT in output.result
    """
    if not await _ollama_is_available():
        pytest.skip("Ollama not running at localhost:11434")

    agent = PlannerAgent()
    output = await agent.execute(_make_input())

    # If fallback triggered, confidence == 0.3 and planner_error is present
    assert output.confidence > 0.5, (
        f"Planner fell back to single-task plan (confidence={output.confidence}). "
        f"Error: {output.result.get('planner_error', 'unknown')}"
    )
    assert len(output.task_plan) >= 1
    assert "planner_error" not in output.result


@pytest.mark.integration
async def test_planner_chatollama_invokes_with_message_list() -> None:
    """ChatOllama.ainvoke() receives a list of messages (not a plain string).

    Verifies the system/user message separation is correctly implemented.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from langchain_core.messages import AIMessage, BaseMessage
    import json

    valid_json = json.dumps([
        {
            "task_id": "task_1",
            "description": "Retrieve and analyze compliance documents for IFRS 15",
            "assigned_agent": "analyst",
            "dependencies": [],
        }
    ])

    with patch("consilium.agents.planner.ChatOllama") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value=AIMessage(content=valid_json))
        mock_cls.return_value = mock_instance

        agent = PlannerAgent()
        await agent.execute(_make_input())

    # Verify ainvoke was called with a list of messages
    call_args = mock_instance.ainvoke.call_args
    messages = call_args[0][0]  # First positional arg
    assert isinstance(messages, list), "ainvoke must receive a list of messages"
    assert len(messages) == 2, "Must pass exactly [SystemMessage, HumanMessage]"

    # Verify message types
    from langchain_core.messages import HumanMessage, SystemMessage
    assert isinstance(messages[0], SystemMessage), "First message must be SystemMessage"
    assert isinstance(messages[1], HumanMessage), "Second message must be HumanMessage"
