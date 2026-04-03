"""
FastAPI application for Consilium workflow execution.

Phase 0: Single /workflow endpoint, Analyst agent only.
Phase 1+: Full multi-agent orchestration via LangGraph.
"""

from fastapi import FastAPI, HTTPException
from consilium.schemas.workflow import WorkflowRequest, WorkflowResponse
from consilium.agents.analyst import AnalystAgent, AnalystInput
from consilium.integrations.retrieval_mock import MockRetrieval
from consilium.config import settings
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Consilium API",
    description="Multi-agent compliance automation system",
    version="0.1.0"
)

# Initialize components at startup
retrieval = MockRetrieval()
analyst = AnalystAgent()


@app.get("/")
async def root() -> dict:
    """Health check."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "phase": "Phase 0: Foundation & Contracts"
    }


@app.get("/health")
async def health_check() -> dict:
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "components": {
            "analyst_agent": analyst.name,
            "retrieval": "MockRetrieval",
            "llm_provider": settings.llm_provider
        }
    }


@app.post("/workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Execute compliance workflow.

    Phase 0: Single-agent execution (Analyst only, mock retrieval).
    Phase 1+: Multi-agent orchestration with LangGraph.

    Args:
        request: WorkflowRequest with compliance query

    Returns:
        WorkflowResponse with analysis results

    Raises:
        HTTPException 500: If workflow execution fails
    """
    try:
        logger.info(f"Executing workflow for query: {request.query[:80]}...")

        # Step 1: Retrieve relevant documents (mock in Phase 0)
        chunks = await retrieval.retrieve(request.query, top_k=3)
        logger.info(f"Retrieved {len(chunks)} chunks")

        # Step 2: Run Analyst agent
        analyst_input = AnalystInput(
            task=f"Analyze compliance risks for: {request.query}",
            context={},
            retrieved_chunks=[chunk.model_dump() for chunk in chunks]
        )

        analyst_output = await analyst.execute(analyst_input)
        logger.info(f"Analyst completed with confidence: {analyst_output.confidence}")

        # Step 3: Build response
        return WorkflowResponse(
            query=request.query,
            result={
                "risk_findings": analyst_output.risk_findings,
                "summary": analyst_output.result.get("summary", "")
            },
            confidence=analyst_output.confidence,
            agents_invoked=[analyst.name]
        )

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )
