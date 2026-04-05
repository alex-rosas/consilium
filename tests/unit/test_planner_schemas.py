"""
Unit tests for PlannerAgent schemas: SubTask, PlannerInput, PlannerOutput.

These test schema validation only — no LLM calls, no agent logic.
"""

import pytest
from pydantic import ValidationError

from consilium.agents.planner import PlannerInput, PlannerOutput, SubTask


class TestSubTaskSchema:
    def test_valid_subtask(self) -> None:
        """SubTask creates successfully with all required fields."""
        task = SubTask(
            task_id="task_1",
            description="Retrieve and analyze revenue recognition documents",
            assigned_agent="analyst",
        )
        assert task.task_id == "task_1"
        assert task.dependencies == []

    def test_subtask_rejects_extra_fields(self) -> None:
        """extra='forbid' prevents unknown keys."""
        with pytest.raises(ValidationError):
            SubTask(
                task_id="task_1",
                description="x" * 20,
                assigned_agent="analyst",
                unknown_field="fail",
            )

    def test_subtask_description_min_length(self) -> None:
        """Description must be at least 10 characters."""
        with pytest.raises(ValidationError):
            SubTask(task_id="task_1", description="short", assigned_agent="analyst")

    def test_subtask_description_max_length(self) -> None:
        """Description must not exceed 500 characters."""
        with pytest.raises(ValidationError):
            SubTask(task_id="task_1", description="x" * 501, assigned_agent="analyst")

    def test_subtask_with_dependencies(self) -> None:
        """Dependencies list is accepted when provided."""
        task = SubTask(
            task_id="task_2",
            description="Synthesize compliance findings into final report",
            assigned_agent="synthesizer",
            dependencies=["task_1"],
        )
        assert task.dependencies == ["task_1"]


class TestPlannerInputSchema:
    def test_valid_planner_input(self) -> None:
        """PlannerInput accepts a valid user_query."""
        inp = PlannerInput(
            task="Decompose compliance query",
            context={},
            user_query="Assess JPMorgan Q3 revenue recognition compliance with IFRS 15",
        )
        assert len(inp.user_query) >= 10

    def test_planner_input_rejects_short_query(self) -> None:
        """user_query must be at least 10 characters."""
        with pytest.raises(ValidationError):
            PlannerInput(task="Decompose", context={}, user_query="short")

    def test_planner_input_rejects_extra_fields(self) -> None:
        """extra='forbid' is inherited from AgentInput."""
        with pytest.raises(ValidationError):
            PlannerInput(
                task="Decompose",
                context={},
                user_query="Valid long enough query here",
                injected="nope",
            )


class TestPlannerOutputSchema:
    def _make_task(self, n: int) -> SubTask:
        return SubTask(
            task_id=f"task_{n}",
            description=f"Task {n} description that is long enough",
            assigned_agent="analyst",
        )

    def test_valid_planner_output_single_task(self) -> None:
        """PlannerOutput accepts a single-task plan."""
        out = PlannerOutput(
            task_plan=[self._make_task(1)],
            result={"plan_summary": "Single task"},
            confidence=0.85,
            reasoning="Fallback plan",
        )
        assert len(out.task_plan) == 1

    def test_valid_planner_output_three_tasks(self) -> None:
        """PlannerOutput accepts up to 3 tasks."""
        out = PlannerOutput(
            task_plan=[self._make_task(i) for i in range(1, 4)],
            result={"plan_summary": "Three tasks"},
            confidence=0.9,
            reasoning="Full decomposition",
        )
        assert len(out.task_plan) == 3

    def test_planner_output_rejects_more_than_3_tasks(self) -> None:
        """PlannerOutput must reject task_plan with more than 3 items."""
        with pytest.raises(ValidationError):
            PlannerOutput(
                task_plan=[self._make_task(i) for i in range(1, 5)],
                result={},
                confidence=0.9,
                reasoning="Too many tasks",
            )

    def test_planner_output_rejects_empty_task_plan(self) -> None:
        """task_plan must have at least 1 task."""
        with pytest.raises(ValidationError):
            PlannerOutput(
                task_plan=[],
                result={},
                confidence=0.9,
                reasoning="Empty plan",
            )

    def test_planner_output_rejects_invalid_dependencies(self) -> None:
        """Dependencies referencing non-existent task_ids are rejected."""
        tasks = [
            SubTask(
                task_id="task_1",
                description="First task description here",
                assigned_agent="analyst",
            ),
            SubTask(
                task_id="task_2",
                description="Second task description here",
                assigned_agent="synthesizer",
                dependencies=["task_999"],  # does not exist
            ),
        ]
        with pytest.raises(ValidationError, match="Invalid dependencies"):
            PlannerOutput(
                task_plan=tasks,
                result={},
                confidence=0.9,
                reasoning="Bad deps",
            )

    def test_planner_output_accepts_valid_dependencies(self) -> None:
        """Cross-task dependencies referencing real task_ids are accepted."""
        tasks = [
            SubTask(
                task_id="task_1",
                description="First task description here",
                assigned_agent="analyst",
            ),
            SubTask(
                task_id="task_2",
                description="Second task description here",
                assigned_agent="synthesizer",
                dependencies=["task_1"],
            ),
        ]
        out = PlannerOutput(
            task_plan=tasks,
            result={"plan_summary": "Two tasks with dependency"},
            confidence=0.85,
            reasoning="Valid dependency chain",
        )
        assert out.task_plan[1].dependencies == ["task_1"]

    def test_planner_output_rejects_extra_fields(self) -> None:
        """extra='forbid' is enforced on PlannerOutput."""
        with pytest.raises(ValidationError):
            PlannerOutput(
                task_plan=[self._make_task(1)],
                result={},
                confidence=0.5,
                reasoning="Test",
                secret="nope",
            )
