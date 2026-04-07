"""Integration tests for API trace_id field in WorkflowResponse."""
import pytest

from consilium.schemas.workflow import WorkflowResponse, WorkflowResult


def test_workflow_response_accepts_trace_id() -> None:
    """WorkflowResponse schema accepts a trace_id string for Phoenix correlation."""
    resp = WorkflowResponse(
        query="What are the IFRS 15 compliance requirements?",
        result=WorkflowResult(risk_findings=[], summary="Analysis complete"),
        confidence=0.85,
        agents_invoked=["planner", "analyst", "synthesizer"],
        trace_id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    )
    assert resp.trace_id == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
    assert len(resp.trace_id) == 32


def test_workflow_response_trace_id_defaults_to_none() -> None:
    """trace_id defaults to None when not provided (e.g., tracing not initialized)."""
    resp = WorkflowResponse(
        query="What are the IFRS 15 compliance requirements?",
        result=WorkflowResult(risk_findings=[], summary="Analysis complete"),
        confidence=0.85,
        agents_invoked=["planner", "analyst", "synthesizer"],
    )
    assert resp.trace_id is None
