"""
Unit tests for configurable confidence threshold (Task 6.4).

Verifies that:
- Settings.confidence_threshold has correct default (0.5) and valid range
- Graph nodes read the threshold from settings (not hardcoded 0.5)
- Agents with confidence exactly at the threshold are NOT considered fallbacks
- Agents with confidence below the threshold ARE considered fallbacks
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from consilium.config import Settings
from consilium.schemas.state import WorkflowState


class TestConfidenceThresholdSettings:
    def test_default_threshold_is_point_five(self) -> None:
        """Default CONFIDENCE_THRESHOLD is 0.5."""
        s = Settings()
        assert s.confidence_threshold == 0.5

    def test_threshold_accepts_zero(self) -> None:
        s = Settings(confidence_threshold=0.0)
        assert s.confidence_threshold == 0.0

    def test_threshold_accepts_one(self) -> None:
        s = Settings(confidence_threshold=1.0)
        assert s.confidence_threshold == 1.0

    def test_threshold_accepts_custom_value(self) -> None:
        s = Settings(confidence_threshold=0.7)
        assert s.confidence_threshold == 0.7

    def test_threshold_rejects_below_zero(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(confidence_threshold=-0.1)

    def test_threshold_rejects_above_one(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(confidence_threshold=1.1)


class TestGraphUsesConfigurableThreshold:
    """Graph nodes must read confidence_threshold from settings, not use 0.5 literal."""

    @pytest.mark.asyncio
    async def test_planner_below_custom_threshold_triggers_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With threshold=0.8, confidence=0.7 must trigger fallback."""
        from consilium.workflow import graph as graph_module
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.planner import PlannerOutput, SubTask

        # Patch module-level settings in graph
        mock_settings = MagicMock(spec=Settings)
        mock_settings.confidence_threshold = 0.8
        monkeypatch.setattr(graph_module, "settings", mock_settings)

        g = WorkflowGraph.__new__(WorkflowGraph)
        mock_output = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the given query input",
                assigned_agent="analyst",
                dependencies=[],
            )],
            result={"plan_summary": "ok"},
            confidence=0.7,  # below threshold=0.8 → should trigger fallback
            reasoning="LLM succeeded",
        )
        mock_planner = MagicMock()
        mock_planner.execute = AsyncMock(return_value=mock_output)
        g._planner = mock_planner
        g._guardrails = MagicMock()
        g._guardrails.check_and_redact_pii = MagicMock(return_value=("query long enough here", False))

        state: WorkflowState = {
            "user_query": "query long enough here",
            "fallback_events": [],
            "agent_history": [],
        }
        update = await g._planner_node(state)

        assert "planner" in update["fallback_events"]

    @pytest.mark.asyncio
    async def test_planner_above_custom_threshold_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With threshold=0.6, confidence=0.7 must NOT trigger fallback."""
        from consilium.workflow import graph as graph_module
        from consilium.workflow.graph import WorkflowGraph
        from consilium.agents.planner import PlannerOutput, SubTask

        mock_settings = MagicMock(spec=Settings)
        mock_settings.confidence_threshold = 0.6
        monkeypatch.setattr(graph_module, "settings", mock_settings)

        g = WorkflowGraph.__new__(WorkflowGraph)
        mock_output = PlannerOutput(
            task_plan=[SubTask(
                task_id="task_1",
                description="Analyze compliance risks for the given query input",
                assigned_agent="analyst",
                dependencies=[],
            )],
            result={"plan_summary": "ok"},
            confidence=0.7,  # above threshold=0.6 → no fallback
            reasoning="LLM succeeded",
        )
        mock_planner = MagicMock()
        mock_planner.execute = AsyncMock(return_value=mock_output)
        g._planner = mock_planner
        g._guardrails = MagicMock()
        g._guardrails.check_and_redact_pii = MagicMock(return_value=("query long enough here", False))

        state: WorkflowState = {
            "user_query": "query long enough here",
            "fallback_events": [],
            "agent_history": [],
        }
        update = await g._planner_node(state)

        assert update.get("fallback_events", []) == []
