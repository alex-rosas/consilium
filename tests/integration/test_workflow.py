"""
Integration tests for the full LangGraph 3-agent workflow.

These tests exercise the complete pipeline:
  PlannerAgent (mocked LLM) → AnalystAgent → SynthesizerAgent

No real Ollama connection is needed — the PlannerAgent LLM is patched
via monkeypatching at the graph level.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from consilium.integrations.retrieval_mock import MockRetrieval
from consilium.workflow.graph import WorkflowGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_planner_llm_response(tasks: List[Dict[str, Any]] | None = None) -> str:
    """Build a valid JSON string for the Planner LLM mock."""
    if tasks is None:
        tasks = [
            {
                "task_id": "task_1",
                "description": "Retrieve and analyze revenue recognition compliance documents",
                "assigned_agent": "analyst",
                "dependencies": [],
            },
            {
                "task_id": "task_2",
                "description": "Synthesize findings into a compliance report",
                "assigned_agent": "synthesizer",
                "dependencies": ["task_1"],
            },
        ]
    return json.dumps(tasks)


def _patch_planner_llm(workflow: WorkflowGraph, response: str | Exception) -> None:
    """Replace the LLM on the PlannerAgent inside a WorkflowGraph."""
    mock_llm = MagicMock()
    if isinstance(response, Exception):
        mock_llm.ainvoke = AsyncMock(side_effect=response)
    else:
        mock_llm.ainvoke = AsyncMock(return_value=response)
    workflow._planner.llm = mock_llm  # type: ignore[attr-defined]


async def _get_mock_chunks() -> List[Dict[str, Any]]:
    retrieval = MockRetrieval()
    chunks = await retrieval.retrieve("test query", top_k=3)
    return [c.model_dump() for c in chunks]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullWorkflowExecution:
    async def test_full_3_agent_path(self) -> None:
        """Planner → Analyst → Synthesizer all execute in order."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert "planner" in state["agent_history"]
        assert "analyst" in state["agent_history"]
        assert "synthesizer" in state["agent_history"]
        assert state["agent_history"] == ["planner", "analyst", "synthesizer"]

    async def test_final_report_is_produced(self) -> None:
        """final_report is present and non-trivially long."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert state.get("final_report") is not None
        assert len(state["final_report"]) >= 50  # type: ignore[arg-type]

    async def test_task_plan_is_in_state(self) -> None:
        """task_plan is written to state by PlannerAgent."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert state.get("task_plan") is not None
        assert len(state["task_plan"]) >= 1  # type: ignore[arg-type]

    async def test_risk_findings_in_state(self) -> None:
        """risk_findings are written to state by AnalystAgent."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert state.get("risk_findings") is not None
        assert len(state["risk_findings"]) > 0  # type: ignore[arg-type]

    async def test_no_error_in_happy_path(self) -> None:
        """Happy path produces no error in state."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert state.get("error") is None


class TestPiiRedactionInWorkflow:
    async def test_workflow_completes_with_pii_in_query(self) -> None:
        """Workflow redacts PII and still completes the full pipeline."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Send report for john.doe@example.com about IFRS 15 compliance review",
            retrieved_chunks=chunks,
        )

        # Workflow should complete despite PII in query
        assert "planner" in state.get("agent_history", [])
        assert state.get("error") is None


class TestPlannerFallbackPath:
    async def test_llm_failure_still_routes_to_analyst(self) -> None:
        """When Planner LLM fails, fallback plan still routes to analyst."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, Exception("Ollama connection refused"))
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        # Fallback plan assigns task to analyst → analyst should still run
        assert "planner" in state.get("agent_history", [])
        assert "analyst" in state.get("agent_history", [])

    async def test_malformed_llm_json_still_completes(self) -> None:
        """Malformed LLM JSON triggers fallback and workflow still completes."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, "this is not json")
        chunks = await _get_mock_chunks()

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=chunks,
        )

        assert "planner" in state.get("agent_history", [])
        assert state.get("task_plan") is not None


class TestEmptyChunksPath:
    async def test_empty_chunks_routes_to_end_after_analyst(self) -> None:
        """With no retrieved chunks, analyst finds no findings → skips synthesizer."""
        workflow = WorkflowGraph()
        _patch_planner_llm(workflow, _make_planner_llm_response())

        state = await workflow.execute(
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
            retrieved_chunks=[],  # No chunks
        )

        # Analyst ran but found nothing → synthesizer should NOT be in history
        assert "planner" in state.get("agent_history", [])
        assert "analyst" in state.get("agent_history", [])
        # No findings → route_after_analyst returns "end"
        assert state.get("final_report") is None
