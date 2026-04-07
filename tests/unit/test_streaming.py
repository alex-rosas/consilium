"""
Unit tests for WorkflowGraph.astream() and SSE event generation.

Tests the astream() generator method in isolation by mocking the compiled
LangGraph graph's astream() to yield pre-canned node updates.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from consilium.schemas.workflow import (
    AnalystCompleteEvent,
    AnalystFindingEvent,
    DoneEvent,
    PlannerCompleteEvent,
    ReportCompleteEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLANNER_UPDATE = {
    "task_plan": [
        {
            "task_id": "task_1",
            "description": "Analyze IFRS 15 revenue recognition risks",
            "assigned_agent": "analyst",
            "dependencies": [],
        }
    ],
    "planner_confidence": 0.85,
    "fallback_events": [],
    "current_agent": "planner",
    "agent_history": ["planner"],
}

_ANALYST_UPDATE = {
    "risk_findings": [
        {
            "clause_reference": "IFRS 15.31",
            "finding": "Revenue recognition timing deviates from IFRS 15 guidance on performance obligations.",
            "risk_level": "Medium",
            "document_source": "IFRS_15",
        }
    ],
    "analyst_confidence": 0.85,
    "fallback_events": [],
    "current_agent": "analyst",
    "agent_history": ["planner", "analyst"],
}

_SYNTHESIZER_UPDATE = {
    "final_report": "## Executive Summary\nOne Medium-risk finding identified.",
    "synthesizer_confidence": 0.85,
    "fallback_events": [],
    "current_agent": "synthesizer",
    "agent_history": ["planner", "analyst", "synthesizer"],
}


async def _mock_graph_astream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
    """Simulate LangGraph astream() yielding one dict per node."""
    yield {"planner": _PLANNER_UPDATE}
    yield {"analyst": _ANALYST_UPDATE}
    yield {"synthesizer": _SYNTHESIZER_UPDATE}


def _make_streaming_workflow() -> Any:
    """Create a WorkflowGraph with the compiled graph's astream mocked."""
    from consilium.workflow.graph import WorkflowGraph

    wf = WorkflowGraph()
    wf._graph = MagicMock()
    wf._graph.astream = _mock_graph_astream
    return wf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkflowGraphAstream:
    async def test_astream_yields_planner_complete(self) -> None:
        """astream must yield a planner_complete event after the planner node."""
        wf = _make_streaming_workflow()
        events = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        planner_events = [e for e in events if e.get("type") == "planner_complete"]
        assert len(planner_events) == 1
        assert planner_events[0]["task_count"] == 1
        assert planner_events[0]["confidence"] == 0.85

    async def test_astream_yields_analyst_finding_per_finding(self) -> None:
        """astream must emit one analyst_finding event per ComplianceFinding."""
        wf = _make_streaming_workflow()
        events = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        finding_events = [e for e in events if e.get("type") == "analyst_finding"]
        assert len(finding_events) == 1
        assert finding_events[0]["finding"]["clause_reference"] == "IFRS 15.31"

    async def test_astream_yields_analyst_complete(self) -> None:
        """astream must yield analyst_complete after all finding events."""
        wf = _make_streaming_workflow()
        events = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        complete_events = [e for e in events if e.get("type") == "analyst_complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["findings_count"] == 1

    async def test_astream_yields_report_complete(self) -> None:
        """astream must yield report_complete with the full report."""
        wf = _make_streaming_workflow()
        events = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        report_events = [e for e in events if e.get("type") == "report_complete"]
        assert len(report_events) == 1
        assert "Executive Summary" in report_events[0]["report"]

    async def test_astream_yields_done_as_final_event(self) -> None:
        """astream must yield done as the very last event."""
        wf = _make_streaming_workflow()
        events = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        assert events[-1]["type"] == "done"

    async def test_astream_event_order(self) -> None:
        """Events must appear in correct pipeline order."""
        wf = _make_streaming_workflow()
        event_types = []
        async for sse_line in wf.astream("test query", []):
            if sse_line.startswith("data: "):
                event_types.append(json.loads(sse_line[6:])["type"])

        # planner_complete before analyst_finding before analyst_complete before report_complete before done
        planner_idx = event_types.index("planner_complete")
        finding_idx = event_types.index("analyst_finding")
        complete_idx = event_types.index("analyst_complete")
        report_idx = event_types.index("report_complete")
        done_idx = event_types.index("done")

        assert planner_idx < finding_idx < complete_idx < report_idx < done_idx

    async def test_astream_sse_format(self) -> None:
        """Each SSE line must be in 'data: {...}\\n\\n' format."""
        wf = _make_streaming_workflow()
        lines = []
        async for sse_line in wf.astream("test query", []):
            lines.append(sse_line)

        for line in lines:
            assert line.startswith("data: "), f"Bad SSE format: {line!r}"
            assert line.endswith("\n\n"), f"Bad SSE terminator: {line!r}"
            json.loads(line[6:])  # Must be valid JSON


class TestStreamEventSchemas:
    def test_planner_complete_valid(self) -> None:
        e = PlannerCompleteEvent(task_count=2, confidence=0.85)
        assert e.type == "planner_complete"

    def test_analyst_finding_valid(self) -> None:
        e = AnalystFindingEvent(finding={"clause_reference": "IFRS 15.31", "risk_level": "High"})
        assert e.type == "analyst_finding"

    def test_analyst_complete_valid(self) -> None:
        e = AnalystCompleteEvent(findings_count=3, confidence=0.7)
        assert e.type == "analyst_complete"

    def test_report_complete_valid(self) -> None:
        e = ReportCompleteEvent(report="## Report")
        assert e.type == "report_complete"

    def test_done_event_valid(self) -> None:
        e = DoneEvent(trace_id="abc123")
        assert e.type == "done"
        assert e.trace_id == "abc123"

    def test_done_event_no_trace_id(self) -> None:
        e = DoneEvent()
        assert e.trace_id is None

    def test_all_events_forbid_extra_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlannerCompleteEvent(task_count=1, confidence=0.5, extra="nope")
