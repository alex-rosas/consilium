"""
Base agent interface.
All agents inherit from BaseAgent and must implement execute() and capabilities.
"""

from abc import ABC, abstractmethod
from consilium.schemas.agent import AgentInput, AgentOutput
from typing import List


class BaseAgent(ABC):
    """
    Abstract base class for all Consilium agents.

    Contract:
    - All agents implement execute() with schema validation
    - All agents declare their capabilities for routing
    - Agents never call each other directly — LangGraph manages flow
    """

    @abstractmethod
    async def execute(self, input: AgentInput) -> AgentOutput:
        """
        Execute agent task with schema validation.

        Args:
            input: Validated agent input (Pydantic model)

        Returns:
            Validated agent output (Pydantic model)

        Raises:
            ValidationError: If input or output schema is violated
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """
        Task types this agent can handle.
        Used by the LangGraph router to determine which agent to invoke.

        Returns:
            List of capability strings, e.g. ["risk_classification", "compliance_analysis"]
        """
        pass

    @property
    def name(self) -> str:
        """Agent name for logging, routing, and response metadata."""
        return self.__class__.__name__
