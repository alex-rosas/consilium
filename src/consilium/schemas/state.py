"""
LangGraph workflow state schema.
Single source of truth for data passed between agents.
This TypedDict is the object LangGraph passes through the graph.
"""

from typing import TypedDict, List, Optional, Dict, Any


class WorkflowState(TypedDict, total=False):
    """
    Shared state across all agents in the workflow.

    Agents read from and write to this state only.
    They never call each other directly.
    LangGraph manages all transitions.
    """

    # Input (set at workflow start)
    user_query: str

    # Planner output (Phase 1)
    task_plan: Optional[List[Dict[str, str]]]

    # Retrieval output (mock in Phase 0-1, real in Phase 2)
    retrieved_chunks: Optional[List[Dict[str, Any]]]

    # Analyst output
    risk_findings: Optional[List[Dict[str, Any]]]

    # Synthesizer output (Phase 1)
    final_report: Optional[str]

    # Routing metadata
    current_agent: str
    agent_history: List[str]

    # Error handling
    error: Optional[str]
