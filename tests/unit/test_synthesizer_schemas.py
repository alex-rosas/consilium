"""
Unit tests for SynthesizerAgent schemas: RiskFinding, SynthesizerInput, SynthesizerOutput.
"""

import pytest
from pydantic import ValidationError

from consilium.agents.synthesizer import RiskFinding, SynthesizerInput, SynthesizerOutput


class TestRiskFindingSchema:
    def test_valid_risk_finding(self) -> None:
        """RiskFinding accepts all required fields."""
        finding = RiskFinding(
            clause_reference="IFRS 15 §31",
            risk_level="High",
            finding="Revenue recognition timing issue identified in Q3 report",
        )
        assert finding.clause_reference == "IFRS 15 §31"
        assert finding.risk_level == "High"

    def test_risk_finding_rejects_extra_fields(self) -> None:
        """extra='forbid' prevents unknown keys."""
        with pytest.raises(ValidationError):
            RiskFinding(
                clause_reference="IFRS 15",
                risk_level="Low",
                finding="Some finding text here",
                extra_key="nope",
            )

    def test_risk_finding_min_length_enforced(self) -> None:
        """finding must be at least 10 characters."""
        with pytest.raises(ValidationError):
            RiskFinding(clause_reference="IFRS 15", risk_level="Low", finding="short")

    def test_all_risk_levels_accepted(self) -> None:
        """High, Medium, Low, and N/A are all valid risk levels."""
        for level in ("High", "Medium", "Low", "N/A"):
            f = RiskFinding(
                clause_reference="Test §1",
                risk_level=level,
                finding="Finding text that is long enough",
            )
            assert f.risk_level == level


class TestSynthesizerInputSchema:
    def _make_finding(self) -> RiskFinding:
        return RiskFinding(
            clause_reference="IFRS 15 §31",
            risk_level="Medium",
            finding="Revenue recognition timing issue",
        )

    def test_valid_synthesizer_input(self) -> None:
        """SynthesizerInput accepts valid findings and original_query."""
        inp = SynthesizerInput(
            task="Synthesize report",
            context={},
            risk_findings=[self._make_finding()],
            original_query="Assess JPMorgan Q3 compliance",
        )
        assert len(inp.risk_findings) == 1

    def test_synthesizer_input_requires_at_least_one_finding(self) -> None:
        """risk_findings must have at least 1 item."""
        with pytest.raises(ValidationError):
            SynthesizerInput(
                task="Synthesize",
                context={},
                risk_findings=[],
                original_query="Some valid query here",
            )

    def test_synthesizer_input_rejects_extra_fields(self) -> None:
        """extra='forbid' is enforced."""
        with pytest.raises(ValidationError):
            SynthesizerInput(
                task="Synthesize",
                context={},
                risk_findings=[self._make_finding()],
                original_query="Valid query",
                injected="nope",
            )


class TestSynthesizerOutputSchema:
    def test_valid_synthesizer_output(self) -> None:
        """SynthesizerOutput accepts a report of at least 50 characters."""
        out = SynthesizerOutput(
            final_report="# Compliance Report\n\nThis is a sufficiently long report body.",
            result={"summary": "No high-risk issues"},
            confidence=0.9,
            reasoning="Rule-based synthesis",
        )
        assert len(out.final_report) >= 50

    def test_synthesizer_output_rejects_short_report(self) -> None:
        """final_report must be at least 50 characters."""
        with pytest.raises(ValidationError):
            SynthesizerOutput(
                final_report="Too short",
                result={},
                confidence=0.9,
                reasoning="test",
            )

    def test_synthesizer_output_rejects_extra_fields(self) -> None:
        """extra='forbid' is enforced."""
        with pytest.raises(ValidationError):
            SynthesizerOutput(
                final_report="x" * 60,
                result={},
                confidence=0.5,
                reasoning="test",
                secret="nope",
            )
