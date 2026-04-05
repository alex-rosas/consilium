"""
FastAPI application for Consilium workflow execution.

Phase 1: Full multi-agent orchestration via LangGraph
         (PlannerAgent → AnalystAgent → SynthesizerAgent)
Phase 0: Was single-agent (Analyst only, mock retrieval)
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException

from consilium.config import settings
from consilium.integrations.retrieval_mock import MockRetrieval
from consilium.schemas.workflow import WorkflowRequest, WorkflowResponse, WorkflowResult
from consilium.workflow.graph import WorkflowGraph

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Consilium API",
    description="Multi-agent compliance automation system",
    version="0.2.0",  # Phase 1
)

# Initialise once at startup — graph compilation and model loading happen here
_workflow = WorkflowGraph()
_retrieval = MockRetrieval()


@app.get("/")
async def root() -> dict:
    """Health check."""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "phase": "Phase 1: Multi-agent orchestration via LangGraph",
    }


@app.get("/health")
async def health_check() -> dict:
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "components": {
            "workflow": "WorkflowGraph (planner→analyst→synthesizer)",
            "retrieval": "MockRetrieval",
            "llm_provider": settings.llm_provider,
            "ollama_model": settings.ollama_model,
        },
    }


@app.post("/workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Execute compliance workflow via LangGraph multi-agent orchestration.

    Phase 1 flow:
    1. Retrieve relevant document chunks (mock)
    2. LangGraph: PlannerAgent decomposes query (Ollama LLM)
    3. LangGraph: AnalystAgent classifies risk from retrieved chunks
    4. LangGraph: SynthesizerAgent builds compliance report
    5. Return structured response

    Args:
        request: WorkflowRequest with compliance query

    Returns:
        WorkflowResponse with risk findings, final report, and agent history

    Raises:
        HTTPException 500: If workflow execution fails
    """
    try:
        logger.info("Executing Phase 1 workflow for query: %.80s...", request.query)

        # Step 1: Retrieve document chunks (mock in Phase 1)
        chunks = await _retrieval.retrieve(request.query, top_k=5)
        chunks_dicts = [c.model_dump() for c in chunks]
        logger.info("Retrieved %d chunks", len(chunks_dicts))

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

        # Build summary from report or fallback
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
            confidence=0.75,  # Fixed in Phase 1; per-agent confidence tracked in Phase 3 eval
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
