"""
Unit tests for SynthesizerAgent logic (rule-based, no LLM).

Phase 2: Uses ComplianceFinding (shared domain model) instead of RiskFinding.
"""

import pytest

from consilium.agents.synthesizer import SynthesizerAgent, SynthesizerInput
from consilium.schemas.findings import ComplianceFinding


def _make_finding(
    clause: str = "IFRS 15 §31",
    risk_level: str = "High",
    finding: str = "Revenue recognition timing issue identified",
    document_source: str = "JPMorgan 10-K Q3 2023",
) -> ComplianceFinding:
    return ComplianceFinding(
        clause_reference=clause,
        risk_level=risk_level,  # type: ignore[arg-type]
        finding=finding,
        document_source=document_source,
    )


def _make_input(
    findings: list[ComplianceFinding],
    query: str = "Assess compliance",
) -> SynthesizerInput:
    return SynthesizerInput(
        task="Synthesize report",
        context={},
        risk_findings=findings,
        original_query=query,
    )


class TestSynthesizerAgentBasic:
    async def test_produces_report_with_correct_counts(self) -> None:
        """Report section headers reflect the actual finding counts."""
        findings = [
            _make_finding(risk_level="High"),
            _make_finding(clause="PCAOB AS 2810", risk_level="Medium",
                          finding="Disclosure gap identified in notes"),
        ]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings))

        assert len(out.final_report) >= 50
        assert "High risk: 1" in out.final_report
        assert "Medium risk: 1" in out.final_report
        assert out.result["high_risk_count"] == "1"
        assert out.result["risk_count"] == "2"
        assert out.confidence > 0.5

    async def test_no_high_risk_summary(self) -> None:
        """Summary sentence differs when there are no High-risk findings."""
        findings = [_make_finding(risk_level="Low", finding="Minor formatting issue noted here")]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings))

        assert "no high-risk issues" in out.result["summary"].lower()

    async def test_high_risk_summary_mentions_count(self) -> None:
        """Summary sentence mentions high-risk count when present."""
        findings = [
            _make_finding(risk_level="High"),
            _make_finding(clause="PCAOB", risk_level="High",
                          finding="Material weakness in controls"),
        ]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings))

        assert "2" in out.result["summary"]

    async def test_multiple_same_risk_level_grouped(self) -> None:
        """Multiple findings of the same level are grouped and counted correctly."""
        findings = [
            _make_finding(clause=f"Clause {i}", risk_level="High",
                          finding=f"High risk finding number {i}")
            for i in range(3)
        ]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings))

        assert out.result["high_risk_count"] == "3"
        assert "High risk: 3" in out.final_report

    async def test_report_includes_original_query(self) -> None:
        """Final report mentions the original query."""
        query = "Assess JPMorgan Q3 compliance"
        findings = [_make_finding()]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings, query=query))

        assert query in out.final_report

    async def test_capabilities_declared(self) -> None:
        """SynthesizerAgent declares expected capabilities."""
        agent = SynthesizerAgent()
        assert "report_synthesis" in agent.capabilities

    async def test_na_findings_handled(self) -> None:
        """N/A risk level findings are included in total count."""
        findings = [
            _make_finding(risk_level="N/A", finding="No compliance issue found for this clause"),
        ]
        agent = SynthesizerAgent()
        out = await agent.execute(_make_input(findings))

        assert out.result["risk_count"] == "1"
        assert out.result["high_risk_count"] == "0"
