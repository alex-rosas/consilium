"""
Unit tests for PlannerAgent logic.

All LLM calls are mocked — no Ollama required.
ChatOllama.ainvoke() returns an AIMessage; mocks reflect this.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from consilium.agents.planner import PlannerAgent, PlannerInput


def _make_agent_with_mock_llm(llm_response: str | Exception) -> PlannerAgent:
    """Build a PlannerAgent whose LLM is replaced with a mock.

    ChatOllama.ainvoke() returns AIMessage — mock returns AIMessage(content=...)
    so that execute() can call response.content correctly.
    """
    agent = PlannerAgent.__new__(PlannerAgent)  # skip __init__ (avoids Ollama connection)
    mock_llm = MagicMock()
    if isinstance(llm_response, Exception):
        mock_llm.ainvoke = AsyncMock(side_effect=llm_response)
    else:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=llm_response))
    agent.llm = mock_llm
    return agent


def _make_input(query: str = "Assess JPMorgan Q3 revenue recognition compliance with IFRS 15") -> PlannerInput:
    return PlannerInput(task="Decompose compliance query", context={}, user_query=query)


class TestPlannerAgentSuccessPath:
    async def test_successful_two_task_decomposition(self) -> None:
        """Planner returns two tasks when LLM gives valid JSON."""
        llm_json = json.dumps([
            {
                "task_id": "task_1",
                "description": "Retrieve and analyze revenue recognition documents",
                "assigned_agent": "analyst",
                "dependencies": [],
            },
            {
                "task_id": "task_2",
                "description": "Synthesize compliance findings into final report",
                "assigned_agent": "synthesizer",
                "dependencies": ["task_1"],
            },
        ])

        agent = _make_agent_with_mock_llm(llm_json)
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 2
        assert output.task_plan[0].task_id == "task_1"
        assert output.task_plan[1].task_id == "task_2"
        assert output.task_plan[1].dependencies == ["task_1"]
        assert output.confidence > 0.5
        assert "plan_summary" in output.result

    async def test_successful_single_task(self) -> None:
        """Planner accepts a single-task response from the LLM."""
        llm_json = json.dumps([
            {
                "task_id": "task_1",
                "description": "Analyze all compliance aspects of the given query",
                "assigned_agent": "analyst",
                "dependencies": [],
            }
        ])

        agent = _make_agent_with_mock_llm(llm_json)
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 1
        assert output.confidence > 0.5

    async def test_capabilities_declared(self) -> None:
        """PlannerAgent declares expected capabilities."""
        agent = _make_agent_with_mock_llm("")
        assert "task_planning" in agent.capabilities
        assert "query_decomposition" in agent.capabilities

    async def test_name_is_class_name(self) -> None:
        """Agent name property returns class name."""
        agent = _make_agent_with_mock_llm("")
        assert agent.name == "PlannerAgent"


class TestPlannerAgentTruncation:
    async def test_truncates_llm_response_to_3_tasks(self) -> None:
        """When LLM returns >3 tasks, Planner silently truncates to 3."""
        tasks = [
            {
                "task_id": f"task_{i}",
                "description": f"Task {i} description that is long enough",
                "assigned_agent": "analyst",
                "dependencies": [],
            }
            for i in range(1, 6)  # 5 tasks
        ]
        agent = _make_agent_with_mock_llm(json.dumps(tasks))
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 3

    async def test_strips_markdown_fences_from_llm_response(self) -> None:
        """Planner handles LLM responses wrapped in ```json ... ``` fences."""
        raw = [
            {
                "task_id": "task_1",
                "description": "Retrieve and analyze compliance documentation",
                "assigned_agent": "analyst",
                "dependencies": [],
            }
        ]
        fenced = f"```json\n{json.dumps(raw)}\n```"

        agent = _make_agent_with_mock_llm(fenced)
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 1
        assert output.task_plan[0].task_id == "task_1"


class TestPlannerAgentFallback:
    async def test_llm_connection_failure_returns_fallback(self) -> None:
        """Network/connection error triggers single-task fallback."""
        agent = _make_agent_with_mock_llm(Exception("Ollama connection refused"))
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 1
        assert output.confidence < 0.5
        assert "planner_error" in output.result

    async def test_malformed_json_returns_fallback(self) -> None:
        """Non-JSON LLM response triggers fallback."""
        agent = _make_agent_with_mock_llm("This is not valid JSON at all.")
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 1
        assert output.confidence < 0.5
        assert "planner_error" in output.result

    async def test_non_list_json_returns_fallback(self) -> None:
        """LLM returning a JSON object (not array) triggers fallback."""
        agent = _make_agent_with_mock_llm('{"task_id": "task_1"}')
        output = await agent.execute(_make_input())

        assert len(output.task_plan) == 1
        assert "planner_error" in output.result

    async def test_fallback_task_assigned_to_analyst(self) -> None:
        """Fallback plan always assigns to 'analyst'."""
        agent = _make_agent_with_mock_llm(Exception("timeout"))
        output = await agent.execute(_make_input())

        assert output.task_plan[0].assigned_agent == "analyst"
