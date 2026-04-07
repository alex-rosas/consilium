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

import hashlib
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List

from langgraph.graph import END, StateGraph
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

from consilium.config import settings
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

    async def astream(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        trace_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream workflow execution as Server-Sent Events.

        Yields SSE-formatted strings as each agent node completes.
        Event sequence: planner_complete → analyst_finding(×N) → analyst_complete
                        → report_complete → done.

        Args:
            user_query: Compliance question from the user.
            retrieved_chunks: List of RetrievalResult dicts.
            trace_id: OTel trace ID to include in the done event.

        Yields:
            SSE lines in the format ``data: {json}\\n\\n``.
        """
        initial_state: WorkflowState = {
            "user_query": user_query,
            "retrieved_chunks": retrieved_chunks,
            "agent_history": [],
        }

        async for chunk in self._graph.astream(initial_state):
            for node_name, update in chunk.items():
                if node_name == "planner":
                    event = {
                        "type": "planner_complete",
                        "task_count": len(update.get("task_plan") or []),
                        "confidence": update.get("planner_confidence", 0.0),
                    }
                    yield f"data: {json.dumps(event)}\n\n"

                elif node_name == "analyst":
                    findings = update.get("risk_findings") or []
                    for finding in findings:
                        yield f"data: {json.dumps({'type': 'analyst_finding', 'finding': finding})}\n\n"
                    event = {
                        "type": "analyst_complete",
                        "findings_count": len(findings),
                        "confidence": update.get("analyst_confidence", 0.0),
                    }
                    yield f"data: {json.dumps(event)}\n\n"

                elif node_name == "synthesizer":
                    event = {
                        "type": "report_complete",
                        "report": update.get("final_report", ""),
                    }
                    yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id})}\n\n"

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
        Writes task_plan to state. Wrapped in an OTel span for observability.
        """
        history: List[str] = list(state.get("agent_history") or [])
        query_hash = hashlib.sha256(state["user_query"].encode()).hexdigest()[:8]

        with tracer.start_as_current_span("planner_node") as span:
            span.set_attribute("agent.name", "planner")
            span.set_attribute("workflow.query_hash", query_hash)
            start_time = time.time()

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

                confidence = output.confidence
                fallback_triggered = confidence < settings.confidence_threshold
                duration_ms = (time.time() - start_time) * 1000

                span.set_attribute("agent.confidence", confidence)
                span.set_attribute("agent.fallback_triggered", fallback_triggered)
                span.set_attribute("agent.duration_ms", duration_ms)

                logger.info(
                    "PlannerAgent produced %d task(s): %s",
                    len(task_plan_dicts),
                    output.result.get("plan_summary", ""),
                )

                fallback_events: List[str] = list(state.get("fallback_events") or [])
                if fallback_triggered:
                    fallback_events.append("planner")
                    logger.warning("Planner fallback activated for query: %.80s", user_query)

                return {
                    "task_plan": task_plan_dicts,
                    "planner_confidence": confidence,
                    "fallback_events": fallback_events,
                    "current_agent": "planner",
                    "agent_history": history + ["planner"],
                }

            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("agent.error", str(exc))
                span.set_attribute("agent.confidence", 0.0)
                span.set_attribute("agent.duration_ms", duration_ms)
                logger.error("PlannerAgent node failed: %s", exc, exc_info=True)
                fallback_events = list(state.get("fallback_events") or [])
                fallback_events.append("planner")
                return {
                    "error": f"Planner error: {exc}",
                    "planner_confidence": 0.0,
                    "fallback_events": fallback_events,
                    "current_agent": "planner",
                    "agent_history": history + ["planner"],
                }

    async def _analyst_node(self, state: WorkflowState) -> _StateUpdate:
        """
        Run AnalystAgent.

        Reads retrieved_chunks from state (injected by API layer).
        Writes risk_findings to state as a list of dicts. Wrapped in an OTel span.
        """
        history: List[str] = list(state.get("agent_history") or [])
        query_hash = hashlib.sha256(state["user_query"].encode()).hexdigest()[:8]

        with tracer.start_as_current_span("analyst_node") as span:
            span.set_attribute("agent.name", "analyst")
            span.set_attribute("workflow.query_hash", query_hash)
            start_time = time.time()

            try:
                raw_chunks: List[Dict[str, Any]] = state.get("retrieved_chunks") or []
                retrieved_chunks = [RetrievalResult.model_validate(c) for c in raw_chunks]

                # Build task context string from Planner output (may be None if Planner failed)
                task_plan_dicts: List[Dict[str, Any]] = state.get("task_plan") or []
                task_context: str | None = None
                if task_plan_dicts:
                    task_context = "\n".join(
                        f"{t.get('task_id', '?')}: {t.get('description', '')}"
                        for t in task_plan_dicts
                    )

                analyst_input = AnalystInput(
                    task="Classify compliance risk for the retrieved document chunks",
                    context={},
                    retrieved_chunks=retrieved_chunks,
                    task_context=task_context,
                )
                output = await self._analyst.execute(analyst_input)

                # AnalystOutput.risk_findings is List[ComplianceFinding] — serialise to dicts
                risk_findings: List[Dict[str, Any]] = [
                    f.model_dump() for f in output.risk_findings
                ]

                confidence = output.confidence
                fallback_triggered = confidence < settings.confidence_threshold
                duration_ms = (time.time() - start_time) * 1000

                span.set_attribute("agent.confidence", confidence)
                span.set_attribute("agent.fallback_triggered", fallback_triggered)
                span.set_attribute("agent.duration_ms", duration_ms)

                logger.info("AnalystAgent produced %d finding(s)", len(risk_findings))

                fallback_events: List[str] = list(state.get("fallback_events") or [])
                if fallback_triggered:
                    fallback_events.append("analyst")
                    logger.warning("Analyst fallback activated for query: %.80s", state.get("user_query", ""))

                return {
                    "risk_findings": risk_findings,
                    "analyst_confidence": confidence,
                    "fallback_events": fallback_events,
                    "current_agent": "analyst",
                    "agent_history": history + ["analyst"],
                }

            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("agent.error", str(exc))
                span.set_attribute("agent.confidence", 0.0)
                span.set_attribute("agent.duration_ms", duration_ms)
                logger.error("AnalystAgent node failed: %s", exc, exc_info=True)
                fallback_events = list(state.get("fallback_events") or [])
                fallback_events.append("analyst")
                return {
                    "error": f"Analyst error: {exc}",
                    "analyst_confidence": 0.0,
                    "fallback_events": fallback_events,
                    "current_agent": "analyst",
                    "agent_history": history + ["analyst"],
                }

    async def _synthesizer_node(self, state: WorkflowState) -> _StateUpdate:
        """
        Run SynthesizerAgent.

        Reads risk_findings from state and validates them as ComplianceFinding models.
        Writes final_report to state. Wrapped in an OTel span for observability.
        """
        history: List[str] = list(state.get("agent_history") or [])
        query_hash = hashlib.sha256(state["user_query"].encode()).hexdigest()[:8]

        with tracer.start_as_current_span("synthesizer_node") as span:
            span.set_attribute("agent.name", "synthesizer")
            span.set_attribute("workflow.query_hash", query_hash)
            start_time = time.time()

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

                confidence = output.confidence
                fallback_triggered = confidence < settings.confidence_threshold
                duration_ms = (time.time() - start_time) * 1000

                span.set_attribute("agent.confidence", confidence)
                span.set_attribute("agent.fallback_triggered", fallback_triggered)
                span.set_attribute("agent.duration_ms", duration_ms)

                logger.info(
                    "SynthesizerAgent produced report (%d chars): %s",
                    len(output.final_report),
                    output.result.get("summary", ""),
                )

                fallback_events: List[str] = list(state.get("fallback_events") or [])
                if fallback_triggered:
                    fallback_events.append("synthesizer")
                    logger.warning("Synthesizer fallback activated for query: %.80s", state.get("user_query", ""))

                return {
                    "final_report": output.final_report,
                    "synthesizer_confidence": confidence,
                    "fallback_events": fallback_events,
                    "current_agent": "synthesizer",
                    "agent_history": history + ["synthesizer"],
                }

            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("agent.error", str(exc))
                span.set_attribute("agent.confidence", 0.0)
                span.set_attribute("agent.duration_ms", duration_ms)
                logger.error("SynthesizerAgent node failed: %s", exc, exc_info=True)
                fallback_events = list(state.get("fallback_events") or [])
                fallback_events.append("synthesizer")
                return {
                    "error": f"Synthesizer error: {exc}",
                    "synthesizer_confidence": 0.0,
                    "fallback_events": fallback_events,
                    "current_agent": "synthesizer",
                    "agent_history": history + ["synthesizer"],
                }
