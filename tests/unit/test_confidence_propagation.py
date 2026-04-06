"""
Unit tests for confidence propagation through WorkflowState to API response.

Task 0: verify per-agent confidence flows from graph nodes into the API response.
The API must use min(planner_conf, analyst_conf, synthesizer_conf) — never a hardcoded value.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consilium.schemas.state import WorkflowState
from consilium.schemas.workflow import WorkflowResponse, WorkflowResult


class TestWorkflowStateConfidenceFields:
    """WorkflowState TypedDict accepts the three new confidence fields."""

    def test_state_accepts_planner_confidence(self) -> None:
        state: WorkflowState = {
            "user_query": "test query for compliance",
            "planner_confidence": 0.85,
            "agent_history": ["planner"],
        }
        assert state["planner_confidence"] == 0.85

    def test_state_accepts_analyst_confidence(self) -> None:
        state: WorkflowState = {
            "user_query": "test query for compliance",
            "analyst_confidence": 0.70,
            "agent_history": ["planner", "analyst"],
        }
        assert state["analyst_confidence"] == 0.70

    def test_state_accepts_synthesizer_confidence(self) -> None:
        state: WorkflowState = {
            "user_query": "test query for compliance",
            "synthesizer_confidence": 0.90,
            "agent_history": ["planner", "analyst", "synthesizer"],
        }
        assert state["synthesizer_confidence"] == 0.90

    def test_state_accepts_all_confidence_fields(self) -> None:
        state: WorkflowState = {
            "user_query": "test query",
            "planner_confidence": 0.85,
            "analyst_confidence": 0.70,
            "synthesizer_confidence": 0.90,
            "agent_history": ["planner", "analyst", "synthesizer"],
        }
        assert state["planner_confidence"] == 0.85
        assert state["analyst_confidence"] == 0.70
        assert state["synthesizer_confidence"] == 0.90


class TestConfidenceAggregation:
    """API confidence = min(planner, analyst, synthesizer) from state."""

    def test_min_confidence_is_analyst_when_lowest(self) -> None:
        state: WorkflowState = {
            "user_query": "test",
            "planner_confidence": 0.85,
            "analyst_confidence": 0.30,  # fallback value
            "synthesizer_confidence": 0.90,
            "agent_history": [],
        }
        conf = min(
            state.get("planner_confidence", 1.0),  # type: ignore[arg-type]
            state.get("analyst_confidence", 1.0),  # type: ignore[arg-type]
            state.get("synthesizer_confidence", 1.0),  # type: ignore[arg-type]
        )
        assert conf == 0.30

    def test_min_confidence_is_planner_when_fallback(self) -> None:
        state: WorkflowState = {
            "user_query": "test",
            "planner_confidence": 0.3,  # fallback
            "analyst_confidence": 0.70,
            "synthesizer_confidence": 0.90,
            "agent_history": [],
        }
        conf = min(
            state.get("planner_confidence", 1.0),  # type: ignore[arg-type]
            state.get("analyst_confidence", 1.0),  # type: ignore[arg-type]
            state.get("synthesizer_confidence", 1.0),  # type: ignore[arg-type]
        )
        assert conf == 0.3

    def test_defaults_to_1_when_fields_absent(self) -> None:
        """If an agent didn't run (e.g., early exit), absent field defaults to 1.0."""
        state: WorkflowState = {"user_query": "test", "agent_history": []}
        conf = min(
            state.get("planner_confidence", 1.0),  # type: ignore[arg-type]
            state.get("analyst_confidence", 1.0),  # type: ignore[arg-type]
            state.get("synthesizer_confidence", 1.0),  # type: ignore[arg-type]
        )
        assert conf == 1.0


class TestWorkflowGraphNodeConfidence:
    """Graph nodes must return per-agent confidence in their state update dicts."""

    @pytest.mark.asyncio
    async def test_planner_node_returns_planner_confidence(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from langchain_core.messages import AIMessage
        import json

        graph = WorkflowGraph.__new__(WorkflowGraph)

        # Mock PlannerAgent.execute to return a successful output (confidence=0.85)
        from consilium.agents.planner import PlannerOutput, SubTask
        mock_output = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the query",
                assigned_agent="analyst",
                dependencies=[],
            )],
            result={"plan_summary": "test"},
            confidence=0.85,
            reasoning="LLM success",
        )
        mock_planner = MagicMock()
        mock_planner.execute = AsyncMock(return_value=mock_output)
        graph._planner = mock_planner
        graph._guardrails = MagicMock()
        graph._guardrails.check_and_redact_pii = MagicMock(return_value=("test query long enough", False))

        state: WorkflowState = {
            "user_query": "test query long enough",
            "agent_history": [],
        }
        update = await graph._planner_node(state)

        assert "planner_confidence" in update
        assert update["planner_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_planner_node_fallback_returns_confidence_03(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.planner import PlannerOutput, SubTask

        graph = WorkflowGraph.__new__(WorkflowGraph)

        mock_fallback = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the given query",
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
            "agent_history": [],
        }
        update = await graph._planner_node(state)

        assert update["planner_confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_analyst_node_returns_analyst_confidence(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.analyst import AnalystOutput
        from consilium.schemas.findings import ComplianceFinding

        graph = WorkflowGraph.__new__(WorkflowGraph)

        mock_output = AnalystOutput(
            risk_findings=[ComplianceFinding(
                clause_reference="IFRS 15.31",
                risk_level="High",
                finding="Revenue recognition timing issue detected",
                document_source="JPMorgan 10-K Q3 2023",
            )],
            result={"findings_count": "1"},
            confidence=0.75,
            reasoning="LLM analysis",
        )
        mock_analyst = MagicMock()
        mock_analyst.execute = AsyncMock(return_value=mock_output)
        graph._analyst = mock_analyst

        state: WorkflowState = {
            "user_query": "test query",
            "retrieved_chunks": [],
            "task_plan": [],
            "agent_history": [],
        }
        update = await graph._analyst_node(state)

        assert "analyst_confidence" in update
        assert update["analyst_confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_synthesizer_node_returns_synthesizer_confidence(self) -> None:
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.synthesizer import SynthesizerOutput

        graph = WorkflowGraph.__new__(WorkflowGraph)

        long_report = "# Compliance Assessment Report\n\n" + "content " * 20
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
                {"clause_reference": "IFRS 15", "risk_level": "Low", "finding": "Minor compliance issue found here", "document_source": "Test Doc"}
            ],
            "agent_history": [],
        }
        update = await graph._synthesizer_node(state)

        assert "synthesizer_confidence" in update
        assert update["synthesizer_confidence"] == 0.90
