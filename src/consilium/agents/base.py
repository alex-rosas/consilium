"""
Base agent interface.
All agents inherit from BaseAgent[I, O] and must implement execute() and capabilities.

Generic parameters:
    I: Agent-specific input type (subclass of AgentInput)
    O: Agent-specific output type (subclass of AgentOutput)

This avoids the Liskov substitution violation that occurs when a subclass
narrows the execute() signature relative to the base class.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from consilium.schemas.agent import AgentInput, AgentOutput

I = TypeVar("I", bound=AgentInput)
O = TypeVar("O", bound=AgentOutput)


class BaseAgent(ABC, Generic[I, O]):
    """
    Abstract base class for all Consilium agents.

    Contract:
    - All agents implement execute() with schema validation
    - All agents declare their capabilities for routing
    - Agents never call each other directly — LangGraph manages flow
    """

    @abstractmethod
    async def execute(self, input: I) -> O:
        """
        Execute agent task with schema validation.

        Args:
            input: Validated agent-specific input (subclass of AgentInput)

        Returns:
            Validated agent-specific output (subclass of AgentOutput)

        Raises:
            ValidationError: If input or output schema is violated
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """
        Task types this agent can handle.
        Used by the LangGraph router to determine which agent to invoke.

        Returns:
            List of capability strings, e.g. ["risk_classification", "compliance_analysis"]
        """
        ...

    @property
    def name(self) -> str:
        """Agent name for logging, routing, and response metadata."""
        return self.__class__.__name__
