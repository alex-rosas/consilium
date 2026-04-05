"""
Unit tests for AnalystAgent.

Phase 2: LLM-based classification via ChatOllama.
- LLM path: mock ChatOllama returns AIMessage(content=json)
- Fallback path: LLM failure → rule-based _classify_risk()
- _classify_risk() still tested directly (it's the fallback)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from consilium.agents.analyst import AnalystAgent, AnalystInput
from consilium.schemas.findings import ComplianceFinding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK = {
    "chunk_text": "IFRS 15.31 requires revenue recognition when performance obligations are met.",
    "metadata": {"section": "IFRS 15.31", "document": "IFRS_15"},
    "score": 0.9,
}

_LLM_FINDING = {
    "clause_reference": "IFRS 15.31",
    "finding": "Revenue recognition timing deviates from IFRS 15 guidance on performance obligations.",
    "risk_level": "Medium",
    "document_source": "IFRS_15",
}


def _make_input(chunks: list | None = None) -> AnalystInput:
    return AnalystInput(
        task="Classify compliance risks",
        context={},
        retrieved_chunks=chunks or [_CHUNK],
    )


def _make_agent_with_mock_llm(llm_response: str | Exception) -> AnalystAgent:
    """Build an AnalystAgent with a mocked LLM (no Ollama connection)."""
    agent = AnalystAgent.__new__(AnalystAgent)
    mock_llm = MagicMock()
    if isinstance(llm_response, Exception):
        mock_llm.ainvoke = AsyncMock(side_effect=llm_response)
    else:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=llm_response))
    agent.llm = mock_llm
    return agent


# ---------------------------------------------------------------------------
# LLM success path
# ---------------------------------------------------------------------------


class TestAnalystAgentLLMPath:
    async def test_llm_path_returns_compliance_findings(self) -> None:
        """Valid LLM JSON produces List[ComplianceFinding] with high confidence."""
        valid_json = json.dumps([_LLM_FINDING])
        agent = _make_agent_with_mock_llm(valid_json)

        output = await agent.execute(_make_input())

        assert len(output.risk_findings) == 1
        assert isinstance(output.risk_findings[0], ComplianceFinding)
        assert output.risk_findings[0].clause_reference == "IFRS 15.31"
        assert output.risk_findings[0].risk_level == "Medium"
        assert output.confidence > 0.5

    async def test_llm_path_multiple_findings(self) -> None:
        """LLM can return multiple findings for multiple chunks."""
        findings = [
            _LLM_FINDING,
            {
                "clause_reference": "PCAOB AS 2810",
                "finding": "Disclosure gap identified in audit notes for Q3 period.",
                "risk_level": "High",
                "document_source": "JPM_10Q_2023Q3",
            },
        ]
        agent = _make_agent_with_mock_llm(json.dumps(findings))
        output = await agent.execute(_make_input())

        assert len(output.risk_findings) == 2
        assert output.risk_findings[1].risk_level == "High"

    async def test_llm_path_empty_findings_accepted(self) -> None:
        """LLM returning empty array is valid (no risks found)."""
        agent = _make_agent_with_mock_llm(json.dumps([]))
        output = await agent.execute(_make_input())

        assert output.risk_findings == []
        assert output.confidence > 0.5  # LLM path succeeded, even with 0 findings

    async def test_llm_invoked_with_message_list(self) -> None:
        """ChatOllama.ainvoke() receives [SystemMessage, HumanMessage]."""
        from langchain_core.messages import HumanMessage, SystemMessage

        agent = _make_agent_with_mock_llm(json.dumps([_LLM_FINDING]))
        await agent.execute(_make_input())

        call_args = agent.llm.ainvoke.call_args
        messages = call_args[0][0]
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    async def test_llm_strips_markdown_fences(self) -> None:
        """LLM JSON wrapped in ```json ... ``` fences is still parsed."""
        fenced = f"```json\n{json.dumps([_LLM_FINDING])}\n```"
        agent = _make_agent_with_mock_llm(fenced)
        output = await agent.execute(_make_input())

        assert len(output.risk_findings) == 1


# ---------------------------------------------------------------------------
# Fallback path (LLM failure → rule-based)
# ---------------------------------------------------------------------------


class TestAnalystAgentFallback:
    async def test_llm_connection_failure_triggers_fallback(self) -> None:
        """Connection error causes rule-based fallback with confidence=0.3."""
        agent = _make_agent_with_mock_llm(Exception("Ollama connection refused"))
        output = await agent.execute(_make_input())

        assert output.confidence < 0.5
        assert len(output.risk_findings) > 0  # rule-based still produces findings
        assert all(isinstance(f, ComplianceFinding) for f in output.risk_findings)

    async def test_malformed_llm_json_triggers_fallback(self) -> None:
        """Non-JSON LLM response triggers rule-based fallback."""
        agent = _make_agent_with_mock_llm("Not valid JSON at all")
        output = await agent.execute(_make_input())

        assert output.confidence < 0.5

    async def test_fallback_findings_are_compliance_findings(self) -> None:
        """Fallback path still produces typed ComplianceFinding objects."""
        agent = _make_agent_with_mock_llm(Exception("timeout"))
        output = await agent.execute(_make_input())

        for f in output.risk_findings:
            assert isinstance(f, ComplianceFinding)
            assert f.risk_level in ("High", "Medium", "Low", "N/A")

    async def test_no_chunks_returns_empty_findings(self) -> None:
        """With no retrieved chunks, analyst returns empty findings."""
        agent = _make_agent_with_mock_llm(json.dumps([]))
        output = await agent.execute(_make_input(chunks=[]))

        assert output.risk_findings == []


# ---------------------------------------------------------------------------
# Rule-based classifier (fallback method — still tested directly)
# ---------------------------------------------------------------------------


class TestAnalystRuleBasedClassifier:
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


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestAnalystSchemas:
    async def test_capabilities_declared(self) -> None:
        agent = AnalystAgent()
        assert "risk_classification" in agent.capabilities
        assert "compliance_analysis" in agent.capabilities

    async def test_name_is_class_name(self) -> None:
        agent = AnalystAgent()
        assert agent.name == "AnalystAgent"

    async def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            AnalystInput(task="test", context={})  # type: ignore

    async def test_malformed_chunk_rejected(self) -> None:
        """A chunk missing required RetrievalResult fields must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnalystInput(
                task="test",
                context={},
                retrieved_chunks=[{"garbage": "data"}],  # type: ignore
            )
