"""
Unit tests for fallback event tracking through WorkflowState to API response.

Task 0b: verify fallback_events list flows from graph nodes through to WorkflowResponse.
An agent appends its name to fallback_events when its confidence < 0.5.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from consilium.schemas.state import WorkflowState
from consilium.schemas.workflow import WorkflowResponse, WorkflowResult


class TestWorkflowStateFallbackEvents:
    """WorkflowState TypedDict accepts the new fallback_events field."""

    def test_state_accepts_empty_fallback_events(self) -> None:
        state: WorkflowState = {
            "user_query": "test query long enough",
            "fallback_events": [],
            "agent_history": [],
        }
        assert state["fallback_events"] == []

    def test_state_accepts_populated_fallback_events(self) -> None:
        state: WorkflowState = {
            "user_query": "test query long enough",
            "fallback_events": ["planner", "analyst"],
            "agent_history": ["planner", "analyst"],
        }
        assert state["fallback_events"] == ["planner", "analyst"]


class TestWorkflowResponseFallbackEvents:
    """WorkflowResponse schema includes fallback_events with a default of []."""

    def test_response_has_empty_fallback_events_by_default(self) -> None:
        resp = WorkflowResponse(
            query="Test query",
            result=WorkflowResult(risk_findings=[], summary="ok"),
            confidence=0.85,
            agents_invoked=["planner", "analyst", "synthesizer"],
        )
        assert resp.fallback_events == []

    def test_response_accepts_fallback_events_list(self) -> None:
        resp = WorkflowResponse(
            query="Test query",
            result=WorkflowResult(risk_findings=[], summary="ok"),
            confidence=0.3,
            agents_invoked=["planner", "analyst", "synthesizer"],
            fallback_events=["planner"],
        )
        assert resp.fallback_events == ["planner"]

    def test_response_rejects_unknown_fields(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            WorkflowResponse(
                query="Test query",
                result=WorkflowResult(risk_findings=[], summary="ok"),
                confidence=0.75,
                agents_invoked=[],
                unknown_field="injected",  # type: ignore
            )


class TestFallbackEventTracking:
    """Graph nodes append agent name to fallback_events when confidence < 0.5."""

    @pytest.mark.asyncio
    async def test_planner_node_appends_to_fallback_events_when_fallback(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.planner import PlannerOutput, SubTask

        graph = WorkflowGraph.__new__(WorkflowGraph)

        # Planner falls back: confidence=0.3
        mock_fallback = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the given query input",
                assigned_agent="analyst",
                dependencies=[],
            )],
            result={"plan_summary": "fallback", "planner_error": "LLM failed"},
            confidence=0.3,
            reasoning="fallback applied",
        )
        mock_planner = MagicMock()
        mock_planner.execute = AsyncMock(return_value=mock_fallback)
        graph._planner = mock_planner
        graph._guardrails = MagicMock()
        graph._guardrails.check_and_redact_pii = MagicMock(return_value=("query long enough here", False))

        state: WorkflowState = {
            "user_query": "query long enough here",
            "fallback_events": [],
            "agent_history": [],
        }
        update = await graph._planner_node(state)

        assert "fallback_events" in update
        assert "planner" in update["fallback_events"]

    @pytest.mark.asyncio
    async def test_planner_node_no_fallback_event_on_success(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.planner import PlannerOutput, SubTask

        graph = WorkflowGraph.__new__(WorkflowGraph)

        mock_output = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the given query input",
                assigned_agent="analyst",
                dependencies=[],
            )],
            result={"plan_summary": "success"},
            confidence=0.85,
            reasoning="LLM succeeded",
        )
        mock_planner = MagicMock()
        mock_planner.execute = AsyncMock(return_value=mock_output)
        graph._planner = mock_planner
        graph._guardrails = MagicMock()
        graph._guardrails.check_and_redact_pii = MagicMock(return_value=("query long enough here", False))

        state: WorkflowState = {
            "user_query": "query long enough here",
            "fallback_events": [],
            "agent_history": [],
        }
        update = await graph._planner_node(state)

        assert update.get("fallback_events", []) == []

    @pytest.mark.asyncio
    async def test_analyst_node_appends_to_fallback_events_when_fallback(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.analyst import AnalystOutput
        from consilium.schemas.findings import ComplianceFinding

        graph = WorkflowGraph.__new__(WorkflowGraph)

        mock_output = AnalystOutput(
            risk_findings=[ComplianceFinding(
                clause_reference="IFRS 15.31",
                risk_level="High",
                finding="Revenue recognition timing issue found here",
                document_source="JPMorgan 10-K Q3 2023",
            )],
            result={"findings_count": "1"},
            confidence=0.3,  # fallback
            reasoning="LLM fallback",
        )
        mock_analyst = MagicMock()
        mock_analyst.execute = AsyncMock(return_value=mock_output)
        graph._analyst = mock_analyst

        state: WorkflowState = {
            "user_query": "test query",
            "retrieved_chunks": [],
            "task_plan": [],
            "fallback_events": ["planner"],  # planner already fell back
            "agent_history": ["planner"],
        }
        update = await graph._analyst_node(state)

        assert "analyst" in update["fallback_events"]
        # Preserves existing fallback events
        assert "planner" in update["fallback_events"]

    @pytest.mark.asyncio
    async def test_synthesizer_node_appends_to_fallback_events_when_fallback(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.synthesizer import SynthesizerOutput

        graph = WorkflowGraph.__new__(WorkflowGraph)

        long_report = "# Compliance Assessment Report\n\n## Executive Summary\n\n" + "content " * 20
        mock_output = SynthesizerOutput(
            final_report=long_report,
            result={"summary": "test", "risk_count": "1", "high_risk_count": "0"},
            confidence=0.3,  # fallback
            reasoning="rule-based fallback",
        )
        mock_synthesizer = MagicMock()
        mock_synthesizer.execute = AsyncMock(return_value=mock_output)
        graph._synthesizer = mock_synthesizer

        state: WorkflowState = {
            "user_query": "test query long enough to pass validation",
            "risk_findings": [
                {"clause_reference": "IFRS 15", "risk_level": "Low",
                 "finding": "Minor compliance issue found here", "document_source": "Test Doc"}
            ],
            "fallback_events": [],
            "agent_history": [],
        }
        update = await graph._synthesizer_node(state)

        assert "synthesizer" in update["fallback_events"]

    @pytest.mark.asyncio
    async def test_fallback_events_empty_when_all_succeed(self) -> None:
        """No fallback events when all agents return confidence >= 0.5."""
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.synthesizer import SynthesizerOutput

        graph = WorkflowGraph.__new__(WorkflowGraph)

        long_report = "# Compliance Assessment Report\n\n## Executive Summary\n\n" + "content " * 20
        mock_output = SynthesizerOutput(
            final_report=long_report,
            result={"summary": "test", "risk_count": "1", "high_risk_count": "0"},
            confidence=0.90,
            reasoning="rule-based synthesis",
        )
        mock_synthesizer = MagicMock()
        mock_synthesizer.execute = AsyncMock(return_value=mock_output)
        graph._synthesizer = mock_synthesizer

        state: WorkflowState = {
            "user_query": "test query long enough to pass validation",
            "risk_findings": [
                {"clause_reference": "IFRS 15", "risk_level": "Low",
                 "finding": "Minor compliance issue found here", "document_source": "Test Doc"}
            ],
            "fallback_events": [],
            "agent_history": [],
        }
        update = await graph._synthesizer_node(state)

        assert update.get("fallback_events", []) == []
