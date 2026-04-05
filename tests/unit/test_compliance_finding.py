"""
Unit tests for ComplianceFinding unified domain schema.

ComplianceFinding is shared between AnalystAgent output and SynthesizerAgent input,
eliminating the fragile field mapping in WorkflowGraph._synthesizer_node().
"""

import pytest
from pydantic import ValidationError

from consilium.schemas.findings import ComplianceFinding


def _valid_finding(**overrides: object) -> ComplianceFinding:
    """Build a valid ComplianceFinding, optionally overriding fields."""
    defaults = {
        "clause_reference": "IFRS 15.31",
        "finding": "Revenue recognition deviates from IFRS 15 guidance",
        "risk_level": "High",
        "document_source": "JPMorgan 10-K Q3 2023",
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return ComplianceFinding(**defaults)  # type: ignore[arg-type]


class TestComplianceFindingValidConstruction:
    def test_valid_finding_accepted(self) -> None:
        """All required fields produces a valid ComplianceFinding."""
        f = _valid_finding()
        assert f.clause_reference == "IFRS 15.31"
        assert f.finding == "Revenue recognition deviates from IFRS 15 guidance"
        assert f.risk_level == "High"
        assert f.document_source == "JPMorgan 10-K Q3 2023"

    def test_all_risk_levels_accepted(self) -> None:
        """High, Medium, Low, and N/A are the only valid risk levels."""
        for level in ("High", "Medium", "Low", "N/A"):
            f = _valid_finding(risk_level=level)
            assert f.risk_level == level

    def test_model_dump_round_trip(self) -> None:
        """ComplianceFinding can be serialised and reconstructed via model_validate."""
        f = _valid_finding()
        data = f.model_dump()
        rebuilt = ComplianceFinding.model_validate(data)
        assert rebuilt == f


class TestComplianceFindingSchemaEnforcement:
    def test_forbids_extra_fields(self) -> None:
        """extra='forbid' rejects unknown keys."""
        with pytest.raises(ValidationError):
            ComplianceFinding(
                clause_reference="IFRS 15.31",
                finding="Revenue recognition issue found here",
                risk_level="High",
                document_source="JPMorgan 10-K",
                unknown_key="not allowed",
            )

    def test_invalid_risk_level_rejected(self) -> None:
        """risk_level must be one of High | Medium | Low | N/A."""
        with pytest.raises(ValidationError):
            _valid_finding(risk_level="Critical")

    def test_empty_risk_level_rejected(self) -> None:
        """Empty string is not a valid risk level."""
        with pytest.raises(ValidationError):
            _valid_finding(risk_level="")


class TestComplianceFindingFieldLengths:
    def test_clause_reference_empty_rejected(self) -> None:
        """clause_reference must have at least 1 character."""
        with pytest.raises(ValidationError):
            _valid_finding(clause_reference="")

    def test_clause_reference_too_long_rejected(self) -> None:
        """clause_reference must not exceed 200 characters."""
        with pytest.raises(ValidationError):
            _valid_finding(clause_reference="x" * 201)

    def test_finding_too_short_rejected(self) -> None:
        """finding must be at least 10 characters."""
        with pytest.raises(ValidationError):
            _valid_finding(finding="short")

    def test_finding_too_long_rejected(self) -> None:
        """finding must not exceed 1000 characters."""
        with pytest.raises(ValidationError):
            _valid_finding(finding="x" * 1001)

    def test_document_source_empty_rejected(self) -> None:
        """document_source must have at least 1 character."""
        with pytest.raises(ValidationError):
            _valid_finding(document_source="")

    def test_document_source_too_long_rejected(self) -> None:
        """document_source must not exceed 500 characters."""
        with pytest.raises(ValidationError):
            _valid_finding(document_source="x" * 501)
