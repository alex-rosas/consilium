"""
FastAPI application for Consilium workflow execution.

Phase 2: Real Quaestor integration via QuaestorClient (configurable).
Phase 1: Was full multi-agent orchestration via LangGraph (mock retrieval).
Phase 0: Was single-agent (Analyst only, mock retrieval).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from opentelemetry import trace

from consilium.config import settings
from consilium.integrations.factory import create_retrieval_client
from consilium.observability.tracing import init_tracing, instrument_fastapi
from consilium.schemas.workflow import WorkflowRequest, WorkflowResponse, WorkflowResult
from consilium.workflow.graph import WorkflowGraph

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — initialised once at startup
# ---------------------------------------------------------------------------

_workflow = WorkflowGraph()
_retrieval = create_retrieval_client(settings)


# ---------------------------------------------------------------------------
# Lifespan: startup health check
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize tracing, log retrieval provider, warn if Quaestor unreachable."""
    import httpx as _httpx

    init_tracing(sample_rate=settings.trace_sample_rate)
    instrument_fastapi(app)
    logger.info(
        "OpenTelemetry tracing initialized (sample_rate=%.2f)", settings.trace_sample_rate
    )

    # Phoenix reachability check — non-blocking; API starts even if Phoenix is down
    try:
        async with _httpx.AsyncClient() as _client:
            await _client.get("http://localhost:6006", timeout=2.0)
        logger.info("Phoenix UI reachable at localhost:6006")
    except Exception:
        logger.warning(
            "Phoenix not reachable at localhost:6006 — traces will not be stored "
            "(API continues normally)"
        )
    logger.info(
        "Consilium API starting. Retrieval provider: %s", settings.retrieval_provider
    )
    if settings.retrieval_provider == "quaestor":
        from consilium.integrations.quaestor_client import QuaestorClient

        if isinstance(_retrieval, QuaestorClient):
            healthy = await _retrieval.health_check()
            if healthy:
                logger.info("Quaestor health check passed at %s", settings.quaestor_base_url)
            else:
                logger.warning(
                    "Quaestor health check FAILED at %s — requests will raise 503",
                    settings.quaestor_base_url,
                )
    yield
    logger.info("Consilium API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Consilium API",
    description="Multi-agent compliance automation system",
    version="0.7.0",  # Phase 7
    lifespan=lifespan,
)


_UI_HTML = Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    """Serve the streaming web UI."""
    return HTMLResponse(content=_UI_HTML.read_text())


@app.get("/health")
async def health_check() -> dict:
    """Detailed health check with component status."""
    retrieval_status = settings.retrieval_provider

    if settings.retrieval_provider == "quaestor":
        from consilium.integrations.quaestor_client import QuaestorClient

        if isinstance(_retrieval, QuaestorClient):
            quaestor_ok = await _retrieval.health_check()
            retrieval_status = f"quaestor ({'healthy' if quaestor_ok else 'unreachable'})"

    return {
        "status": "healthy",
        "version": "0.7.0",
        "components": {
            "workflow": "WorkflowGraph (planner→analyst→synthesizer)",
            "retrieval": retrieval_status,
            "llm_provider": settings.llm_provider,
            "ollama_model": settings.ollama_model,
        },
    }


@app.post("/workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Execute compliance workflow via LangGraph multi-agent orchestration.

    Phase 2 flow:
    1. Retrieve relevant document chunks (MockRetrieval or QuaestorClient per config)
    2. LangGraph: PlannerAgent decomposes query (ChatOllama LLM)
    3. LangGraph: AnalystAgent classifies risk from retrieved chunks
    4. LangGraph: SynthesizerAgent builds compliance report
    5. Return structured response

    Args:
        request: WorkflowRequest with compliance query

    Returns:
        WorkflowResponse with risk findings, final report, and agent history

    Raises:
        HTTPException 503: If Quaestor is unreachable (when using quaestor provider)
        HTTPException 500: If workflow execution fails
    """
    try:
        logger.info("Executing workflow for query: %.80s...", request.query)

        # Extract trace ID from current OTel span (populated by FastAPI instrumentation)
        current_span = trace.get_current_span()
        trace_id: str | None = None
        if current_span.get_span_context().is_valid:
            trace_id = format(current_span.get_span_context().trace_id, "032x")

        # Step 1: Retrieve document chunks (provider controlled by RETRIEVAL_PROVIDER)
        chunks = await _retrieval.retrieve(request.query, top_k=5)
        chunks_dicts = [c.model_dump() for c in chunks]
        logger.info("Retrieved %d chunks via %s", len(chunks_dicts), settings.retrieval_provider)

        # Step 2–4: Run LangGraph (planner → analyst → synthesizer)
        final_state = await _workflow.execute(
            user_query=request.query,
            retrieved_chunks=chunks_dicts,
        )

        # Surface workflow error as HTTP 500
        if final_state.get("error"):
            logger.error("Workflow error in state: %s", final_state["error"])
            raise HTTPException(
                status_code=500,
                detail=f"Workflow error: {final_state['error']}",
            )

        # Extract results from final state
        risk_findings: List[dict] = final_state.get("risk_findings") or []
        final_report: str = final_state.get("final_report") or ""
        agents_invoked: List[str] = final_state.get("agent_history") or []
        fallback_events: List[str] = final_state.get("fallback_events") or []

        summary = final_report[:300] if final_report else "No synthesis report generated"

        # Aggregate confidence: min across all agents that ran (absent fields default to 1.0)
        final_confidence = min(
            final_state.get("planner_confidence", 1.0),
            final_state.get("analyst_confidence", 1.0),
            final_state.get("synthesizer_confidence", 1.0),
        )

        logger.info(
            "Workflow complete. Agents: %s. Findings: %d. Report: %d chars. Confidence: %.2f.",
            agents_invoked,
            len(risk_findings),
            len(final_report),
            final_confidence,
        )

        return WorkflowResponse(
            query=request.query,
            result=WorkflowResult(
                risk_findings=risk_findings,  # type: ignore[arg-type]
                summary=summary,
            ),
            confidence=final_confidence,
            agents_invoked=agents_invoked,
            fallback_events=fallback_events,
            trace_id=trace_id,
            error=None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled workflow exception: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal workflow error: {exc}",
        )


@app.post("/workflow/stream")
async def stream_workflow(request: WorkflowRequest) -> StreamingResponse:
    """
    Stream compliance workflow execution as Server-Sent Events.

    Events are emitted as each agent completes:
      - planner_complete  — task decomposition done
      - analyst_finding   — one per ComplianceFinding (may be 0)
      - analyst_complete  — all findings emitted
      - report_complete   — final report text
      - done              — stream finished, trace_id included

    Args:
        request: WorkflowRequest with compliance query.

    Returns:
        StreamingResponse with media_type text/event-stream.
    """
    # Capture trace ID before entering the generator (span is active here)
    current_span = trace.get_current_span()
    trace_id: str | None = None
    if current_span.get_span_context().is_valid:
        trace_id = format(current_span.get_span_context().trace_id, "032x")

    chunks = await _retrieval.retrieve(request.query, top_k=5)
    chunks_dicts = [c.model_dump() for c in chunks]

    return StreamingResponse(
        _workflow.astream(request.query, chunks_dicts, trace_id=trace_id),
        media_type="text/event-stream",
    )
