"""
LangGraph workflow state schema.
Single source of truth for data passed between agents.

NOTE on Dict[str, Any]: This is the ONLY place in the codebase where Dict[str, Any]
is acceptable. WorkflowState is a TypedDict (required by LangGraph's StateGraph),
which cannot contain Pydantic models. Validation happens at agent boundaries:
  - Graph → Agent: Convert dict to Pydantic via Model.model_validate()
  - Agent → Graph: Convert Pydantic to dict via .model_dump()

Key ownership (which agent writes each field):
  - user_query:        Set by API layer before graph execution
  - task_plan:         Written by PlannerAgent (List of SubTask dicts)
  - retrieved_chunks:  Injected by API layer (mock in Phase 1, Quaestor in Phase 2)
  - risk_findings:     Written by AnalystAgent (List of risk finding dicts)
  - final_report:      Written by SynthesizerAgent
  - current_agent:     Written by each graph node
  - agent_history:     Appended by each graph node
  - error:             Written by any agent/node on failure
"""

from typing import Any, Dict, List, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """
    Shared state across all agents in the workflow.

    total=False allows partial state updates — each node returns only the
    keys it writes, and LangGraph merges them into the accumulated state.

    Agents read from and write to this state only.
    They never call each other directly — LangGraph manages all transitions.
    """

    # Input (set at workflow start by API layer)
    user_query: str

    # Planner output — List of SubTask dicts (see note on Dict[str, Any] above)
    task_plan: Optional[List[Dict[str, Any]]]

    # Retrieval output — List of RetrievalResult dicts
    retrieved_chunks: Optional[List[Dict[str, Any]]]

    # Analyst output — List of risk finding dicts
    risk_findings: Optional[List[Dict[str, Any]]]

    # Synthesizer output
    final_report: Optional[str]

    # Orchestration metadata
    current_agent: str
    agent_history: List[str]

    # Error handling — set by any agent on failure
    error: Optional[str]
