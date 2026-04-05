"""
Unit tests: task context propagation from Planner to Analyst (Task 5).

AnalystInput.task_context is an Optional[str] — a pre-formatted string of
the task plan built by WorkflowGraph._analyst_node() before calling the agent.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from consilium.agents.analyst import AnalystAgent, AnalystInput


_CHUNK = {
    "chunk_text": "IFRS 15.31 requires revenue recognition when performance obligations are met.",
    "metadata": {"section": "IFRS 15.31", "document": "IFRS_15"},
    "score": 0.9,
}

_FINDING = {
    "clause_reference": "IFRS 15.31",
    "finding": "Revenue recognition timing deviates from IFRS 15 guidance.",
    "risk_level": "Medium",
    "document_source": "IFRS_15",
}


def _make_agent_with_mock_llm(llm_response: str | Exception) -> AnalystAgent:
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_llm = MagicMock()
    if isinstance(llm_response, Exception):
        mock_llm.ainvoke = AsyncMock(side_effect=llm_response)
    else:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=llm_response))
    agent.llm = mock_llm
    return agent


class TestAnalystInputTaskContext:
    def test_task_context_is_optional(self) -> None:
        """AnalystInput can be created without task_context."""
        inp = AnalystInput(
            task="Classify risks",
            context={},
            retrieved_chunks=[_CHUNK],
        )
        assert inp.task_context is None

    def test_task_context_accepted_as_string(self) -> None:
        """AnalystInput accepts a non-None task_context string."""
        inp = AnalystInput(
            task="Classify risks",
            context={},
            retrieved_chunks=[_CHUNK],
            task_context="task_1: Analyze IFRS 15 compliance\ntask_2: Synthesize report",
        )
        assert inp.task_context is not None
        assert "task_1" in inp.task_context

    def test_task_context_forbids_extra_fields(self) -> None:
        """extra='forbid' still enforced on AnalystInput."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalystInput(
                task="Classify",
                context={},
                retrieved_chunks=[_CHUNK],
                unknown_field="nope",
            )


class TestAnalystPromptIncludesTaskContext:
    async def test_task_context_appears_in_user_message(self) -> None:
        """When task_context is set, the HumanMessage includes it."""
        task_ctx = "task_1: Analyze IFRS 15 revenue recognition compliance"
        agent = _make_agent_with_mock_llm(json.dumps([_FINDING]))
        await agent.execute(
            AnalystInput(
                task="Classify risks",
                context={},
                retrieved_chunks=[_CHUNK],
                task_context=task_ctx,
            )
        )

        call_args = agent.llm.ainvoke.call_args
        messages = call_args[0][0]
        user_message_content = messages[1].content  # HumanMessage
        assert task_ctx in user_message_content

    async def test_no_task_context_prompt_still_works(self) -> None:
        """When task_context is None, prompt is built normally."""
        agent = _make_agent_with_mock_llm(json.dumps([_FINDING]))
        output = await agent.execute(
            AnalystInput(
                task="Classify risks",
                context={},
                retrieved_chunks=[_CHUNK],
                task_context=None,
            )
        )

        assert len(output.risk_findings) == 1
        # Verify prompt didn't contain "Task plan" section
        call_args = agent.llm.ainvoke.call_args
        messages = call_args[0][0]
        assert "Task plan" not in messages[1].content

    async def test_task_context_included_in_prompt_under_task_plan_header(self) -> None:
        """Task context appears under 'Task plan:' header in user message."""
        task_ctx = "task_1: Analyze revenue recognition"
        agent = _make_agent_with_mock_llm(json.dumps([_FINDING]))
        await agent.execute(
            AnalystInput(
                task="Classify risks",
                context={},
                retrieved_chunks=[_CHUNK],
                task_context=task_ctx,
            )
        )

        call_args = agent.llm.ainvoke.call_args
        user_content = call_args[0][0][1].content
        assert "Task plan:" in user_content
        assert task_ctx in user_content


class TestWorkflowGraphTaskContextPropagation:
    async def test_graph_passes_task_plan_to_analyst(self) -> None:
        """WorkflowGraph._analyst_node passes task_plan from state as task_context string."""
        from unittest.mock import AsyncMock as AM, MagicMock as MM, patch
        from langchain_core.messages import AIMessage

        from consilium.integrations.retrieval_mock import MockRetrieval
        from consilium.workflow.graph import WorkflowGraph

        workflow = WorkflowGraph()

        # Mock planner LLM to return a real two-task plan
        planner_json = json.dumps([
            {
                "task_id": "task_1",
                "description": "Analyze IFRS 15 revenue recognition compliance risks",
                "assigned_agent": "analyst",
                "dependencies": [],
            },
            {
                "task_id": "task_2",
                "description": "Synthesize findings into a compliance report",
                "assigned_agent": "synthesizer",
                "dependencies": ["task_1"],
            },
        ])
        mock_planner_llm = MM()
        mock_planner_llm.ainvoke = AM(return_value=AIMessage(content=planner_json))
        workflow._planner.llm = mock_planner_llm

        # Capture what AnalystInput was called with
        captured_inputs: list[AnalystInput] = []
        original_execute = workflow._analyst.execute

        async def capture_execute(inp: AnalystInput) -> object:
            captured_inputs.append(inp)
            return await original_execute(inp)

        workflow._analyst.execute = capture_execute  # type: ignore[method-assign]

        # Also mock analyst LLM to avoid real Ollama call
        mock_analyst_llm = MM()
        mock_analyst_llm.ainvoke = AM(
            return_value=AIMessage(content=json.dumps([_FINDING]))
        )
        workflow._analyst.llm = mock_analyst_llm

        chunks = [c.model_dump() for c in await MockRetrieval().retrieve("test")]
        await workflow.execute(user_query="Assess IFRS 15 compliance", retrieved_chunks=chunks)

        assert len(captured_inputs) == 1
        inp = captured_inputs[0]
        assert inp.task_context is not None
        assert "task_1" in inp.task_context
        assert "Analyze IFRS 15" in inp.task_context
