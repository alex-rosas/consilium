"""
Unit tests for Pydantic schemas.
Validates that schema validation works correctly at all boundaries.
"""

import pytest
from pydantic import ValidationError
from consilium.schemas.agent import AgentInput, AgentOutput
from consilium.schemas.state import WorkflowState
from consilium.schemas.workflow import WorkflowRequest, WorkflowResponse, WorkflowResult


class TestAgentSchemas:

    def test_agent_input_valid(self) -> None:
        data = AgentInput(task="Test task", context={"key": "value"})
        assert data.task == "Test task"
        assert data.context["key"] == "value"

    def test_agent_input_missing_task_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentInput(context={})  # type: ignore

    def test_agent_input_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentInput(task="test", context={}, unknown_field="value")  # type: ignore

    def test_agent_output_valid(self) -> None:
        output = AgentOutput(result={"key": "val"}, confidence=0.85, reasoning="test")
        assert output.confidence == 0.85

    def test_agent_output_confidence_too_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(result={}, confidence=1.5, reasoning="test")

    def test_agent_output_confidence_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(result={}, confidence=-0.1, reasoning="test")


class TestWorkflowSchemas:

    def test_workflow_request_valid(self) -> None:
        req = WorkflowRequest(query="What are the compliance risks for this query?")
        assert len(req.query) >= 10

    def test_workflow_request_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowRequest(query="Short")

    def test_workflow_request_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowRequest(query="x" * 1001)

    def test_workflow_request_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowRequest(query="What are the compliance risks?", unknown_field="injected")  # type: ignore

    def test_workflow_response_valid(self) -> None:
        resp = WorkflowResponse(
            query="Test query",
            result=WorkflowResult(risk_findings=[], summary="ok"),
            confidence=0.75,
            agents_invoked=["AnalystAgent"]
        )
        assert resp.confidence == 0.75
        assert len(resp.agents_invoked) == 1
        assert resp.error is None

    def test_workflow_response_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowResponse(
                query="Test",
                result=WorkflowResult(risk_findings=[], summary="ok"),
                confidence=0.75,
                agents_invoked=[],
                unknown_field="injected"  # type: ignore
            )


class TestWorkflowState:

    def test_partial_state_with_user_query(self) -> None:
        """WorkflowState (total=False) allows partial construction."""
        state: WorkflowState = {"user_query": "Test query", "current_agent": "planner"}
        assert state["user_query"] == "Test query"

    def test_partial_state_with_error(self) -> None:
        """Error-only partial state is valid."""
        state: WorkflowState = {"error": "LLM timeout", "agent_history": ["planner"]}
        assert state["error"] == "LLM timeout"

    def test_state_accepts_task_plan_dicts(self) -> None:
        """task_plan accepts List[Dict[str, Any]] (SubTask serialized)."""
        state: WorkflowState = {
            "user_query": "Assess compliance",
            "task_plan": [
                {"task_id": "task_1", "description": "Analyze docs", "assigned_agent": "analyst", "dependencies": []}
            ],
            "agent_history": ["planner"],
        }
        assert state["task_plan"] is not None
        assert state["task_plan"][0]["task_id"] == "task_1"

    def test_state_accepts_full_pipeline_output(self) -> None:
        """Full state with all agent outputs is valid."""
        state: WorkflowState = {
            "user_query": "Test",
            "task_plan": [{"task_id": "task_1", "description": "x", "assigned_agent": "analyst", "dependencies": []}],
            "retrieved_chunks": [{"chunk_text": "text", "metadata": {}, "score": 0.9}],
            "risk_findings": [{"clause_reference": "IFRS 15", "risk_level": "High", "finding": "issue"}],
            "final_report": "# Report\n\nContent",
            "current_agent": "synthesizer",
            "agent_history": ["planner", "analyst", "synthesizer"],
            "error": None,
        }
        assert state["agent_history"] == ["planner", "analyst", "synthesizer"]
