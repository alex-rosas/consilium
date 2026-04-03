"""
Unit tests for Pydantic schemas.
Validates that schema validation works correctly at all boundaries.
"""

import pytest
from pydantic import ValidationError
from consilium.schemas.agent import AgentInput, AgentOutput
from consilium.schemas.workflow import WorkflowRequest, WorkflowResponse


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

    def test_workflow_response_valid(self) -> None:
        resp = WorkflowResponse(
            query="Test query",
            result={"findings": []},
            confidence=0.75,
            agents_invoked=["AnalystAgent"]
        )
        assert resp.confidence == 0.75
        assert len(resp.agents_invoked) == 1
        assert resp.error is None
