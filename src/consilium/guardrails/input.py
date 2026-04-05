"""
Input guardrails: PII detection and redaction.

Applied to user_query before PlannerAgent runs.
Uses Presidio for detection (spaCy NLP backend) and AnonymizerEngine for redaction.

Setup requirement: en_core_web_lg spaCy model must be installed.
  uv pip install https://github.com/explosion/spacy-models/releases/download/
      en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
"""

from __future__ import annotations

import logging
from typing import Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# PII entity types to detect — keep minimal and high-confidence in Phase 1
_DETECTED_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
]


class InputGuardrails:
    """
    PII detection and redaction for user queries.

    Instantiation loads the spaCy NLP model (expensive — do once at startup).
    """

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def check_and_redact_pii(self, text: str) -> Tuple[str, bool]:
        """
        Detect and redact PII from input text.

        Args:
            text: Raw user query.

        Returns:
            (redacted_text, pii_detected)
            - redacted_text: Original text with PII replaced by entity-type placeholders.
            - pii_detected: True if any PII was found and redacted.
        """
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=_DETECTED_ENTITIES,
        )

        if not results:
            return text, False

        detected_types = {r.entity_type for r in results}
        logger.warning("PII detected in user query — redacting types: %s", detected_types)

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, True
