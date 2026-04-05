"""
LangGraph StateGraph: multi-agent orchestration for Consilium Phase 1.

Flow:  START → planner → analyst → synthesizer → END
                ↘ (error/no tasks) ↗ (error/no findings) ↘
                             END                        END

Each node:
  1. Reads the relevant slice of WorkflowState
  2. Converts dicts to Pydantic models (validation at boundary)
  3. Calls the agent's execute() method
  4. Converts Pydantic output back to dicts
  5. Returns a *partial* state dict — LangGraph merges it
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from consilium.agents.analyst import AnalystAgent, AnalystInput
from consilium.agents.planner import PlannerAgent, PlannerInput
from consilium.agents.synthesizer import SynthesizerAgent, SynthesizerInput
from consilium.guardrails.input import InputGuardrails
from consilium.integrations.retrieval_mock import RetrievalResult
from consilium.schemas.findings import ComplianceFinding
from consilium.schemas.state import WorkflowState
from consilium.workflow.router import route_after_analyst, route_after_planner

logger = logging.getLogger(__name__)

# Type alias for partial state updates returned by each node
_StateUpdate = Dict[str, Any]


class WorkflowGraph:
    """
    LangGraph state machine for 3-agent compliance workflow.

    Instantiated once at application startup; reused across requests.
    The compiled graph is thread-safe and async-safe.
    """

    def __init__(self) -> None:
        self._planner = PlannerAgent()
        self._analyst = AnalystAgent()
        self._synthesizer = SynthesizerAgent()
        self._guardrails = InputGuardrails()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> WorkflowState:
        """
        Run the full 3-agent workflow and return the final state.

        Args:
            user_query: Compliance question from the user.
            retrieved_chunks: List of RetrievalResult dicts (from mock or Quaestor).

        Returns:
            Final WorkflowState after all nodes have executed.
        """
        initial_state: WorkflowState = {
            "user_query": user_query,
            "retrieved_chunks": retrieved_chunks,
            "agent_history": [],
        }
        final_state: WorkflowState = await self._graph.ainvoke(initial_state)  # type: ignore[assignment]
        return final_state

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow: StateGraph = StateGraph(WorkflowState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            route_after_planner,
            {"analyst": "analyst", "end": END},
        )
        workflow.add_conditional_edges(
            "analyst",
            route_after_analyst,
            {"synthesizer": "synthesizer", "end": END},
        )
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _planner_node(self, state: WorkflowState) -> _StateUpdate:
        """
        Run PlannerAgent.

        Applies PII guardrails to user_query before calling the LLM.
        Writes task_plan to state.
        """
        history: List[str] = list(state.get("agent_history") or [])

        try:
            user_query = state["user_query"]
            redacted_query, pii_detected = self._guardrails.check_and_redact_pii(user_query)
            if pii_detected:
                logger.warning("PII detected and redacted from user_query before PlannerAgent")

            planner_input = PlannerInput(
                task="Decompose the compliance query into sub-tasks",
                context={},
                user_query=redacted_query,
            )
            output = await self._planner.execute(planner_input)
            task_plan_dicts = [task.model_dump() for task in output.task_plan]

            logger.info(
                "PlannerAgent produced %d task(s): %s",
                len(task_plan_dicts),
                output.result.get("plan_summary", ""),
            )

            return {
                "task_plan": task_plan_dicts,
                "current_agent": "planner",
                "agent_history": history + ["planner"],
            }

        except Exception as exc:
            logger.error("PlannerAgent node failed: %s", exc, exc_info=True)
            return {
                "error": f"Planner error: {exc}",
                "current_agent": "planner",
                "agent_history": history + ["planner"],
            }

    async def _analyst_node(self, state: WorkflowState) -> _StateUpdate:
        """
        Run AnalystAgent.

        Reads retrieved_chunks from state (injected by API layer).
        Writes risk_findings to state as a list of dicts.
        """
        history: List[str] = list(state.get("agent_history") or [])

        try:
            raw_chunks: List[Dict[str, Any]] = state.get("retrieved_chunks") or []
            retrieved_chunks = [RetrievalResult.model_validate(c) for c in raw_chunks]

            analyst_input = AnalystInput(
                task="Classify compliance risk for the retrieved document chunks",
                context={},
                retrieved_chunks=retrieved_chunks,
            )
            output = await self._analyst.execute(analyst_input)

            # AnalystOutput.risk_findings is List[ComplianceFinding] — serialise to dicts
            risk_findings: List[Dict[str, Any]] = [
                f.model_dump() for f in output.risk_findings
            ]

            logger.info("AnalystAgent produced %d finding(s)", len(risk_findings))

            return {
                "risk_findings": risk_findings,
                "current_agent": "analyst",
                "agent_history": history + ["analyst"],
            }

        except Exception as exc:
            logger.error("AnalystAgent node failed: %s", exc, exc_info=True)
            return {
                "error": f"Analyst error: {exc}",
                "current_agent": "analyst",
                "agent_history": history + ["analyst"],
            }

    async def _synthesizer_node(self, state: WorkflowState) -> _StateUpdate:
        """
        Run SynthesizerAgent.

        Reads risk_findings from state and validates them as ComplianceFinding models.
        Writes final_report to state.
        """
        history: List[str] = list(state.get("agent_history") or [])

        try:
            raw_findings: List[Dict[str, Any]] = state.get("risk_findings") or []

            # Both Analyst and Synthesizer use ComplianceFinding — direct pass-through.
            risk_findings = [ComplianceFinding.model_validate(f) for f in raw_findings]

            synthesizer_input = SynthesizerInput(
                task="Aggregate risk findings into a structured compliance report",
                context={},
                risk_findings=risk_findings,
                original_query=state["user_query"],
            )
            output = await self._synthesizer.execute(synthesizer_input)

            logger.info(
                "SynthesizerAgent produced report (%d chars): %s",
                len(output.final_report),
                output.result.get("summary", ""),
            )

            return {
                "final_report": output.final_report,
                "current_agent": "synthesizer",
                "agent_history": history + ["synthesizer"],
            }

        except Exception as exc:
            logger.error("SynthesizerAgent node failed: %s", exc, exc_info=True)
            return {
                "error": f"Synthesizer error: {exc}",
                "current_agent": "synthesizer",
                "agent_history": history + ["synthesizer"],
            }
