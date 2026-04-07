"""Unit tests for workflow graph OTel instrumentation."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consilium.schemas.state import WorkflowState
from consilium.workflow.graph import WorkflowGraph


@pytest.fixture
def mock_state() -> WorkflowState:
    return {
        "user_query": "test compliance query long enough",
        "retrieved_chunks": [],
        "task_plan": [],
        "agent_history": [],
        "fallback_events": [],
    }


@pytest.fixture
def graph() -> WorkflowGraph:
    """WorkflowGraph with all agents mocked — no LLM calls."""
    g = WorkflowGraph.__new__(WorkflowGraph)
    g._guardrails = MagicMock()
    g._guardrails.check_and_redact_pii = MagicMock(
        return_value=("test compliance query long enough", False)
    )
    g._planner = MagicMock()
    g._analyst = MagicMock()
    g._synthesizer = MagicMock()
    return g


@pytest.mark.asyncio
async def test_planner_node_creates_span_with_attributes(graph: WorkflowGraph, mock_state: WorkflowState) -> None:
    """Planner node creates OTel span with required attributes."""
    from consilium.agents.planner import PlannerOutput, SubTask

    mock_output = PlannerOutput(
        task_plan=[SubTask(
            task_id="task_1",
            description="Analyze compliance risks for the given query",
            assigned_agent="analyst",
            dependencies=[],
        )],
        result={"plan_summary": "success"},
        confidence=0.85,
        reasoning="LLM succeeded",
    )
    graph._planner.execute = AsyncMock(return_value=mock_output)

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        await graph._planner_node(mock_state)

    mock_tracer.start_as_current_span.assert_called_once_with("planner_node")
    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert calls["agent.name"] == "planner"
    assert "agent.confidence" in calls
    assert "agent.duration_ms" in calls
    assert "workflow.query_hash" in calls


@pytest.mark.asyncio
async def test_analyst_node_creates_span_with_attributes(graph: WorkflowGraph, mock_state: WorkflowState) -> None:
    """Analyst node creates OTel span with required attributes."""
    from consilium.agents.analyst import AnalystOutput

    mock_output = AnalystOutput(
        risk_findings=[],
        result={"summary": "no findings"},
        confidence=0.85,
        reasoning="No risks found",
    )
    graph._analyst.execute = AsyncMock(return_value=mock_output)

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        await graph._analyst_node(mock_state)

    mock_tracer.start_as_current_span.assert_called_once_with("analyst_node")
    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert calls["agent.name"] == "analyst"


@pytest.mark.asyncio
async def test_synthesizer_node_creates_span_with_attributes(graph: WorkflowGraph) -> None:
    """Synthesizer node creates OTel span with required attributes."""
    from consilium.agents.synthesizer import SynthesizerOutput

    long_report = "# Report\n\n## Executive Summary\n\n" + "content " * 20
    mock_output = SynthesizerOutput(
        final_report=long_report,
        result={"summary": "test", "risk_count": "0", "high_risk_count": "0"},
        confidence=0.90,
        reasoning="synthesis done",
    )
    graph._synthesizer.execute = AsyncMock(return_value=mock_output)

    state: WorkflowState = {
        "user_query": "test compliance query long enough",
        "risk_findings": [],
        "agent_history": [],
        "fallback_events": [],
    }

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        await graph._synthesizer_node(state)

    mock_tracer.start_as_current_span.assert_called_once_with("synthesizer_node")
    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert calls["agent.name"] == "synthesizer"


@pytest.mark.asyncio
async def test_node_sets_error_attribute_on_exception(graph: WorkflowGraph, mock_state: WorkflowState) -> None:
    """Planner node sets error attribute when agent execution fails."""
    graph._planner.execute = AsyncMock(side_effect=RuntimeError("LLM unreachable"))

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        result = await graph._planner_node(mock_state)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert "agent.error" in calls
    assert "LLM unreachable" in calls["agent.error"]
    assert result["planner_confidence"] == 0.0


@pytest.mark.asyncio
async def test_fallback_attribute_true_when_confidence_low(graph: WorkflowGraph, mock_state: WorkflowState) -> None:
    """fallback_triggered attribute is True when agent confidence < 0.5."""
    from consilium.agents.planner import PlannerOutput, SubTask

    mock_output = PlannerOutput(
        task_plan=[SubTask(
            task_id="task_1",
            description="Analyze compliance risks for the given query",
            assigned_agent="analyst",
            dependencies=[],
        )],
        result={"plan_summary": "fallback", "planner_error": "failed"},
        confidence=0.3,
        reasoning="fallback applied",
    )
    graph._planner.execute = AsyncMock(return_value=mock_output)

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        await graph._planner_node(mock_state)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert calls["agent.fallback_triggered"] is True


@pytest.mark.asyncio
async def test_fallback_attribute_false_when_confidence_high(graph: WorkflowGraph, mock_state: WorkflowState) -> None:
    """fallback_triggered attribute is False when agent confidence >= 0.5."""
    from consilium.agents.planner import PlannerOutput, SubTask

    mock_output = PlannerOutput(
        task_plan=[SubTask(
            task_id="task_1",
            description="Analyze compliance risks for the given query",
            assigned_agent="analyst",
            dependencies=[],
        )],
        result={"plan_summary": "success"},
        confidence=0.85,
        reasoning="LLM succeeded",
    )
    graph._planner.execute = AsyncMock(return_value=mock_output)

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    with patch("consilium.workflow.graph.tracer") as mock_tracer:
        mock_tracer.start_as_current_span.return_value = mock_span
        await graph._planner_node(mock_state)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert calls["agent.fallback_triggered"] is False
