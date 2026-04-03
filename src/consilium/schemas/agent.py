"""
Base schemas for agent inputs and outputs.
All agents extend these for consistent validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any


class AgentInput(BaseModel):
    """
    Base input schema for all agents.
    Extend this with agent-specific fields.
    """

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields — non-negotiable

    task: str = Field(..., description="The task this agent should perform")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for task execution"
    )


class AgentOutput(BaseModel):
    """
    Base output schema for all agents.
    Extend this with agent-specific result fields.
    """

    model_config = ConfigDict(extra="forbid")

    result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent execution result"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0–1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how this result was produced"
    )
