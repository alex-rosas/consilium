"""
Analyst agent: Risk classification and compliance analysis.

Phase 0: Rule-based keyword classification (validates contract pattern)
Phase 2: risk_findings now typed as List[ComplianceFinding] — unified schema with Synthesizer.
Phase 2+: LLM-powered analysis via Ollama (Task 4).
"""

from __future__ import annotations

from typing import List

from pydantic import ConfigDict, Field

from consilium.agents.base import BaseAgent
from consilium.integrations.retrieval_mock import RetrievalResult
from consilium.schemas.agent import AgentInput, AgentOutput
from consilium.schemas.findings import ComplianceFinding


class AnalystInput(AgentInput):
    """Analyst-specific input — extends base with retrieved chunks."""

    model_config = ConfigDict(extra="forbid")

    retrieved_chunks: List[RetrievalResult] = Field(
        ...,
        description="Retrieved document chunks to analyze",
    )


class AnalystOutput(AgentOutput):
    """Analyst-specific output — extends base with risk findings."""

    model_config = ConfigDict(extra="forbid")

    risk_findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description="Risk classifications per document chunk",
    )


class AnalystAgent(BaseAgent[AnalystInput, AnalystOutput]):
    """
    Risk classification agent.

    Analyzes retrieved compliance documents and classifies risks
    according to regulatory standards.

    Phase 0: Uses rule-based classification (deterministic, no LLM).
    Phase 1+: Will use Ollama LLM with structured prompt.
    """

    def __init__(self) -> None:
        pass

    @property
    def capabilities(self) -> List[str]:
        return ["risk_classification", "compliance_analysis", "regulatory_assessment"]

    async def execute(self, input: AnalystInput) -> AnalystOutput:
        """
        Analyze document chunks and classify compliance risks.

        Args:
            input: AnalystInput with task description and retrieved chunks

        Returns:
            AnalystOutput with per-chunk risk classifications
        """
        chunks = input.retrieved_chunks
        risk_findings: List[ComplianceFinding] = []

        for chunk in chunks:
            risk_level = self._classify_risk(chunk.chunk_text)

            risk_findings.append(
                ComplianceFinding(
                    clause_reference=chunk.metadata.get("section", "Unknown"),
                    risk_level=risk_level,
                    finding=chunk.chunk_text[:100] + "...",
                    document_source=chunk.metadata.get("document", "Unknown"),
                )
            )

        confidence = 0.75  # Fixed placeholder for Phase 0 rule-based classification

        reasoning = (
            f"Analyzed {len(chunks)} document chunks. "
            f"Identified {len(risk_findings)} risk findings via keyword classification."
        )

        return AnalystOutput(
            result={"summary": f"Completed risk analysis of {len(chunks)} chunks"},
            confidence=confidence,
            reasoning=reasoning,
            risk_findings=risk_findings
        )

    def _classify_risk(self, text: str) -> str:
        """
        Rule-based risk classification by keyword matching.

        Phase 0 only. Replaced with LLM call in Phase 1.

        Returns:
            "High" | "Medium" | "Low" | "N/A"
        """
        text_lower = text.lower()

        if any(word in text_lower for word in ["violat", "non-compliance", "material weakness"]):
            return "High"

        if any(word in text_lower for word in ["requires", "must", "shall", "attention"]):
            return "Medium"

        if any(word in text_lower for word in ["optional", "recommended", "guidance"]):
            return "Low"

        return "N/A"
