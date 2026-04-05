"""
SynthesizerAgent: Aggregates risk findings into a structured compliance report.

Phase 1: Rule-based report generation (no LLM).
Phase 2+: Can be enhanced with LLM-based narrative synthesis.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

from consilium.agents.base import BaseAgent
from consilium.schemas.agent import AgentInput, AgentOutput


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RiskFinding(BaseModel):
    """
    Single risk finding produced by AnalystAgent.

    Domain object embedded inside SynthesizerInput — not an agent I/O schema itself.
    """

    model_config = ConfigDict(extra="forbid")

    clause_reference: str = Field(..., description="Regulation clause (e.g., IFRS 15 §31)")
    risk_level: str = Field(..., description="High | Medium | Low | N/A")
    finding: str = Field(..., min_length=10, max_length=1000)


class SynthesizerInput(AgentInput):
    """Input for the SynthesizerAgent."""

    model_config = ConfigDict(extra="forbid")

    risk_findings: List[RiskFinding] = Field(..., min_length=1)
    original_query: str = Field(..., min_length=10, max_length=1000)


class SynthesizerOutput(AgentOutput):
    """Output: structured compliance report."""

    model_config = ConfigDict(extra="forbid")

    final_report: str = Field(..., min_length=50, max_length=5000)


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class SynthesizerAgent(BaseAgent[SynthesizerInput, SynthesizerOutput]):
    """
    Aggregates AnalystAgent findings into a structured markdown compliance report.

    Rule-based in Phase 1: groups findings by risk level, builds report sections.
    No LLM call — high confidence, deterministic output.
    """

    @property
    def capabilities(self) -> List[str]:
        return ["report_synthesis", "aggregation"]

    async def execute(self, input: SynthesizerInput) -> SynthesizerOutput:
        """
        Build a structured compliance report from risk findings.

        Groups findings by risk level (High → Medium → Low → N/A),
        writes executive summary with counts, then detailed per-finding sections.
        """
        findings = input.risk_findings
        by_level = self._group_by_risk_level(findings)

        report_lines: List[str] = [
            "# Compliance Assessment Report",
            "",
            f"**Query:** {input.original_query}",
            "",
            "## Executive Summary",
            f"Total findings: {len(findings)}",
            f"- High risk: {len(by_level['High'])}",
            f"- Medium risk: {len(by_level['Medium'])}",
            f"- Low risk: {len(by_level['Low'])}",
            f"- N/A: {len(by_level['N/A'])}",
            "",
            "## Detailed Findings",
            "",
        ]

        for level in ("High", "Medium", "Low", "N/A"):
            if by_level[level]:
                report_lines.append(f"### {level} Risk")
                for f in by_level[level]:
                    report_lines.append(f"- **{f.clause_reference}:** {f.finding}")
                report_lines.append("")

        final_report = "\n".join(report_lines)

        return SynthesizerOutput(
            final_report=final_report,
            result={
                "summary": self._create_summary(findings),
                "risk_count": str(len(findings)),
                "high_risk_count": str(len(by_level["High"])),
            },
            confidence=0.9,
            reasoning=f"Rule-based synthesis of {len(findings)} finding(s) into structured report",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_by_risk_level(
        self, findings: List[RiskFinding]
    ) -> Dict[str, List[RiskFinding]]:
        groups: Dict[str, List[RiskFinding]] = {
            "High": [],
            "Medium": [],
            "Low": [],
            "N/A": [],
        }
        for f in findings:
            bucket = f.risk_level if f.risk_level in groups else "N/A"
            groups[bucket].append(f)
        return groups

    def _create_summary(self, findings: List[RiskFinding]) -> str:
        high_count = sum(1 for f in findings if f.risk_level == "High")
        if high_count > 0:
            return f"Identified {high_count} high-risk compliance issue(s) requiring immediate attention."
        if findings:
            return f"Identified {len(findings)} compliance finding(s) with no high-risk issues."
        return "No compliance issues identified."
