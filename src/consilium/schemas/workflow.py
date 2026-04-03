"""
API request and response schemas for the /workflow endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class WorkflowRequest(BaseModel):
    """Request schema for workflow execution."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Compliance question or audit request",
        examples=["What are the revenue recognition compliance risks for JPMorgan Q3 2023?"]
    )

    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional workflow configuration"
    )


class WorkflowResponse(BaseModel):
    """Response schema for workflow execution."""

    query: str = Field(..., description="Original query")
    result: Dict[str, Any] = Field(..., description="Workflow execution result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    agents_invoked: List[str] = Field(
        default_factory=list,
        description="Agents called during execution"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
