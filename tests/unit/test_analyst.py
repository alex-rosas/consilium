"""
Unit tests for Analyst agent.
Validates execution, schema compliance, and classification logic.
"""

import pytest
from pydantic import ValidationError
from consilium.agents.analyst import AnalystAgent, AnalystInput


@pytest.mark.asyncio
class TestAnalystAgent:

    async def test_execution_returns_valid_output(self) -> None:
        agent = AnalystAgent()
        input_data = AnalystInput(
            task="Analyze compliance risks",
            context={},
            retrieved_chunks=[
                {
                    "chunk_text": "IFRS 15.31 requires revenue recognition...",
                    "metadata": {"section": "31", "document": "IFRS_15"},
                    "score": 0.9
                }
            ]
        )
        output = await agent.execute(input_data)
        assert 0.0 <= output.confidence <= 1.0
        assert output.reasoning != ""
        assert len(output.risk_findings) > 0

    async def test_capabilities_declared(self) -> None:
        agent = AnalystAgent()
        caps = agent.capabilities
        assert "risk_classification" in caps
        assert "compliance_analysis" in caps

    async def test_name_is_class_name(self) -> None:
        agent = AnalystAgent()
        assert agent.name == "AnalystAgent"

    async def test_risk_classification_high(self) -> None:
        agent = AnalystAgent()
        assert agent._classify_risk("This violates the regulation") == "High"

    async def test_risk_classification_medium(self) -> None:
        agent = AnalystAgent()
        assert agent._classify_risk("The standard requires attention") == "Medium"

    async def test_risk_classification_low(self) -> None:
        agent = AnalystAgent()
        assert agent._classify_risk("Optional guidance is recommended") == "Low"

    async def test_risk_classification_na(self) -> None:
        agent = AnalystAgent()
        assert agent._classify_risk("General information about practices") == "N/A"

    async def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            AnalystInput(task="test", context={})  # type: ignore

    async def test_malformed_chunk_rejected(self) -> None:
        """A chunk missing required RetrievalResult fields must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnalystInput(
                task="test",
                context={},
                retrieved_chunks=[{"garbage": "data"}]  # type: ignore
            )
