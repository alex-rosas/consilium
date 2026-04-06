"""
API request and response schemas for the /workflow endpoint.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict


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
    error: Optional[str] = Field(default=None, description="Error message if failed")
