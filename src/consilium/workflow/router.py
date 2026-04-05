"""
LangGraph conditional routing functions.

Each function reads the current WorkflowState and returns a string
that LangGraph maps to the next node name.
"""

from __future__ import annotations

from typing import Literal

from consilium.schemas.state import WorkflowState


def route_after_planner(
    state: WorkflowState,
) -> Literal["analyst", "end"]:
    """
    Decide next node after PlannerAgent completes.

    Rules (evaluated in order):
    1. Any error → end (fail-fast)
    2. task_plan is empty/missing → end (nothing to do)
    3. At least one task assigned to "analyst" → analyst
    4. Default → end (synthesizer-only plan not supported in Phase 1)
    """
    if state.get("error"):
        return "end"

    task_plan = state.get("task_plan") or []
    if not task_plan:
        return "end"

    for task in task_plan:
        if task.get("assigned_agent") == "analyst":
            return "analyst"

    return "end"


def route_after_analyst(
    state: WorkflowState,
) -> Literal["synthesizer", "end"]:
    """
    Decide next node after AnalystAgent completes.

    Rules:
    1. Any error → end (fail-fast)
    2. risk_findings present → synthesizer
    3. No findings → end
    """
    if state.get("error"):
        return "end"

    risk_findings = state.get("risk_findings") or []
    if risk_findings:
        return "synthesizer"

    return "end"
