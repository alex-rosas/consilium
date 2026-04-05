"""
FastAPI application for Consilium workflow execution.

Phase 2: Real Quaestor integration via QuaestorClient (configurable).
Phase 1: Was full multi-agent orchestration via LangGraph (mock retrieval).
Phase 0: Was single-agent (Analyst only, mock retrieval).
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException

from consilium.config import settings
from consilium.integrations.factory import create_retrieval_client
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
    """Log retrieval provider and warn if Quaestor is unreachable at startup."""
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
    version="0.3.0",  # Phase 2
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict:
    """Root health check."""
    return {
        "status": "healthy",
        "version": "0.3.0",
        "phase": "Phase 2: Real Quaestor integration",
    }


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
        "version": "0.3.0",
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

        summary = final_report[:300] if final_report else "No synthesis report generated"

        logger.info(
            "Workflow complete. Agents: %s. Findings: %d. Report: %d chars.",
            agents_invoked,
            len(risk_findings),
            len(final_report),
        )

        return WorkflowResponse(
            query=request.query,
            result=WorkflowResult(
                risk_findings=risk_findings,  # type: ignore[arg-type]
                summary=summary,
            ),
            confidence=0.75,
            agents_invoked=agents_invoked,
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
