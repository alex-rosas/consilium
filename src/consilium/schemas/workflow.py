"""
API request and response schemas for the /workflow endpoint.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class WorkflowOptions(BaseModel):
    """Optional workflow configuration. Empty in Phase 0 — extensible in Phase 1+."""

    model_config = ConfigDict(extra="forbid")


class WorkflowResult(BaseModel):
    """Structured result from workflow execution."""

    model_config = ConfigDict(extra="forbid")

    risk_findings: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Per-chunk risk classifications from the Analyst agent"
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of the analysis"
    )


class WorkflowRequest(BaseModel):
    """Request schema for workflow execution."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Compliance question or audit request",
        examples=["What are the revenue recognition compliance risks for JPMorgan Q3 2023?"]
    )

    options: Optional[WorkflowOptions] = Field(
        default=None,
        description="Optional workflow configuration"
    )


class WorkflowResponse(BaseModel):
    """Response schema for workflow execution."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Original query")
    result: WorkflowResult = Field(..., description="Structured workflow execution result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    agents_invoked: List[str] = Field(
        default_factory=list,
        description="Agents called during execution"
    )
    fallback_events: List[str] = Field(
        default_factory=list,
        description="Agents that fell back to rule-based logic (confidence < 0.5)"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="OpenTelemetry trace ID for Phoenix correlation"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Streaming event models (POST /workflow/stream)
# ---------------------------------------------------------------------------


class PlannerCompleteEvent(BaseModel):
    """Emitted when PlannerAgent finishes decomposing the query."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["planner_complete"] = "planner_complete"
    task_count: int = Field(..., ge=0, description="Number of sub-tasks produced")
    confidence: float = Field(..., ge=0.0, le=1.0)


class AnalystFindingEvent(BaseModel):
    """Emitted once per ComplianceFinding produced by AnalystAgent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["analyst_finding"] = "analyst_finding"
    finding: Dict[str, Any] = Field(..., description="Single ComplianceFinding dict")


class AnalystCompleteEvent(BaseModel):
    """Emitted when AnalystAgent finishes all risk classifications."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["analyst_complete"] = "analyst_complete"
    findings_count: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ReportCompleteEvent(BaseModel):
    """Emitted when SynthesizerAgent finishes the compliance report."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["report_complete"] = "report_complete"
    report: str = Field(..., description="Full compliance report markdown")


class DoneEvent(BaseModel):
    """Final event — stream is complete."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"
    trace_id: Optional[str] = Field(default=None, description="OTel trace ID")


StreamEvent = Annotated[
    Union[
        PlannerCompleteEvent,
        AnalystFindingEvent,
        AnalystCompleteEvent,
        ReportCompleteEvent,
        DoneEvent,
    ],
    Field(discriminator="type"),
]
