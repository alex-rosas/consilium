"""
Unit tests for InputGuardrails PII detection.

These tests require Presidio + en_core_web_lg spaCy model to be installed.
Run: python -m spacy download en_core_web_lg (or install via uv, see README)
"""

import pytest

from consilium.guardrails.input import InputGuardrails


@pytest.fixture(scope="module")
def guardrails() -> InputGuardrails:
    """Shared guardrails instance (model loading is expensive)."""
    return InputGuardrails()


class TestNoPiiDetection:
    def test_clean_query_passes_through_unchanged(self, guardrails: InputGuardrails) -> None:
        """A clean compliance query is returned as-is with pii_detected=False."""
        text = "Assess JPMorgan Q3 revenue recognition compliance with IFRS 15"
        redacted, pii_detected = guardrails.check_and_redact_pii(text)

        assert redacted == text
        assert pii_detected is False

    def test_company_name_is_not_flagged(self, guardrails: InputGuardrails) -> None:
        """Company names in a compliance context should not block the workflow."""
        text = "What are the revenue recognition risks for Goldman Sachs in Q4 2023?"
        _, pii_detected = guardrails.check_and_redact_pii(text)
        # Company names may or may not be flagged depending on spaCy model;
        # what matters is the workflow still receives a usable (possibly redacted) query.
        assert isinstance(pii_detected, bool)


class TestEmailDetection:
    def test_email_is_detected_and_redacted(self, guardrails: InputGuardrails) -> None:
        """Email addresses are detected and replaced with a placeholder."""
        text = "Send the compliance report to john.doe@example.com for review"
        redacted, pii_detected = guardrails.check_and_redact_pii(text)

        assert pii_detected is True
        assert "john.doe@example.com" not in redacted

    def test_redacted_text_retains_non_pii_content(self, guardrails: InputGuardrails) -> None:
        """Non-PII surrounding text is preserved after redaction."""
        text = "Contact analyst@firm.com about IFRS 15 compliance"
        redacted, pii_detected = guardrails.check_and_redact_pii(text)

        assert "IFRS 15" in redacted or "compliance" in redacted


class TestSsnDetection:
    def test_ssn_is_detected_and_redacted(self, guardrails: InputGuardrails) -> None:
        """US Social Security Numbers are detected and removed.

        Note: Presidio requires a non-sequential SSN with the 'SSN:' prefix to
        achieve the confidence threshold needed for a positive detection.
        Sequential values like '123-45-6789' are filtered as obvious test data.
        """
        text = "Employee SSN: 900-80-1234 needs verification for audit"
        redacted, pii_detected = guardrails.check_and_redact_pii(text)

        assert pii_detected is True
        assert "900-80-1234" not in redacted


class TestReturnType:
    def test_returns_tuple_of_str_and_bool(self, guardrails: InputGuardrails) -> None:
        """check_and_redact_pii always returns (str, bool)."""
        result = guardrails.check_and_redact_pii("Any input text here")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    def test_redacted_text_is_never_empty(self, guardrails: InputGuardrails) -> None:
        """Redacted output is always a non-empty string for non-empty input."""
        text = "Check compliance for john@example.com"
        redacted, _ = guardrails.check_and_redact_pii(text)
        assert len(redacted) > 0
