"""
AnalystAgent: Risk classification and compliance analysis.

Phase 0: Rule-based keyword classification (validates contract pattern).
Phase 2: LLM-based classification via ChatOllama.
         Falls back to rule-based if LLM fails (same confidence=0.3 pattern as Planner).
"""

from __future__ import annotations

import json
import logging
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ConfigDict, Field

from consilium.agents.base import BaseAgent
from consilium.integrations.retrieval_mock import RetrievalResult
from consilium.schemas.agent import AgentInput, AgentOutput
from consilium.schemas.findings import ComplianceFinding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnalystInput(AgentInput):
    """Analyst-specific input — extends base with retrieved chunks."""

    model_config = ConfigDict(extra="forbid")

    retrieved_chunks: List[RetrievalResult] = Field(
        ...,
        description="Retrieved document chunks to analyze",
    )
    task_context: str | None = Field(
        default=None,
        description=(
            "Optional pre-formatted task plan string from PlannerAgent. "
            "When present, the LLM uses it to focus analysis on planned sub-tasks."
        ),
    )


class AnalystOutput(AgentOutput):
    """Analyst-specific output — extends base with risk findings."""

    model_config = ConfigDict(extra="forbid")

    risk_findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description="Risk classifications per document chunk",
    )


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class AnalystAgent(BaseAgent[AnalystInput, AnalystOutput]):
    """
    Risk classification agent.

    Calls ChatOllama to classify compliance risks in retrieved document chunks.
    Returns List[ComplianceFinding] validated against the shared domain schema.

    On any LLM failure (connection error, malformed JSON, schema violation),
    falls back to rule-based keyword classification so the workflow can continue.
    """

    def __init__(self) -> None:
        from consilium.config import settings

        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,
        )

    @property
    def capabilities(self) -> List[str]:
        return ["risk_classification", "compliance_analysis", "regulatory_assessment"]

    async def execute(self, input: AnalystInput) -> AnalystOutput:
        """
        Classify compliance risks across all retrieved chunks in a single LLM call.

        Calls ChatOllama with chunk content and instructs it to output a JSON array
        of ComplianceFinding-compatible objects. Falls back to rule-based classification
        if the LLM fails or returns unparseable output.

        Args:
            input: AnalystInput with retrieved chunks.

        Returns:
            AnalystOutput with risk findings and confidence score.
        """
        chunks = input.retrieved_chunks

        if not chunks:
            return AnalystOutput(
                risk_findings=[],
                result={"summary": "No chunks to analyze"},
                confidence=0.85,
                reasoning="No retrieved chunks — skipping LLM call",
            )

        sys_msg, user_msg = self._build_prompt(chunks, task_context=input.task_context)

        try:
            response = await self.llm.ainvoke([sys_msg, user_msg])
            raw_text = response.content  # AIMessage.content is the string
            findings = self._parse_llm_response(raw_text)

            logger.info("AnalystAgent LLM produced %d finding(s)", len(findings))
            return AnalystOutput(
                risk_findings=findings,
                result={"summary": f"LLM analysis of {len(chunks)} chunks: {len(findings)} finding(s)"},
                confidence=0.85,
                reasoning=f"ChatOllama classified {len(chunks)} chunk(s) into {len(findings)} finding(s)",
            )

        except Exception as exc:
            logger.warning("AnalystAgent LLM failed, using rule-based fallback: %s", exc)
            return self._fallback_findings(chunks, str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        chunks: List[RetrievalResult],
        task_context: str | None = None,
    ) -> tuple[SystemMessage, HumanMessage]:
        """Return (SystemMessage, HumanMessage) for ChatOllama.ainvoke().

        System message sets the role and output schema.
        User message provides the optional task plan context followed by chunk content.
        """
        system_content = (
            "You are a compliance risk analyst specialising in financial regulations.\n"
            "For each document chunk provided, identify compliance risks.\n"
            "Output ONLY a valid JSON array — no markdown fences, no explanation.\n"
            "Schema: "
            '[{"clause_reference": "...", "finding": "...", '
            '"risk_level": "High|Medium|Low|N/A", "document_source": "..."}]\n'
            "Rules:\n"
            "- clause_reference: the specific regulation clause (e.g. 'IFRS 15.31')\n"
            "- finding: 10-500 char evidence + analytical interpretation\n"
            "- risk_level: exactly one of High, Medium, Low, N/A\n"
            "- document_source: the document identifier from the metadata\n"
            "- Output one finding per relevant chunk (skip chunks with no risk)\n"
            "- If no compliance risks found, output an empty array: []"
        )

        parts: List[str] = []

        if task_context:
            parts.append(f"Task plan:\n{task_context}")

        chunk_lines = []
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.metadata.get("document", "Unknown")
            section = chunk.metadata.get("section", "")
            label = f"{doc}" + (f" §{section}" if section else "")
            chunk_lines.append(f"Chunk {i} [{label}]:\n\"{chunk.chunk_text}\"")

        parts.append("Retrieved chunks:\n\n" + "\n\n".join(chunk_lines))
        user_content = "\n\n".join(parts)
        return SystemMessage(content=system_content), HumanMessage(content=user_content)

    def _parse_llm_response(self, response: str) -> List[ComplianceFinding]:
        """Parse and validate the LLM JSON response into ComplianceFinding objects.

        Strips markdown fences if present. Raises ValueError on parse or
        validation failure so the caller can trigger the fallback.
        """
        text = response.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            inner = [ln for ln in lines[1:] if ln.strip() != "```"]
            text = "\n".join(inner).strip()

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError("Expected JSON array of finding objects")

        return [ComplianceFinding.model_validate(item) for item in raw]

    def _fallback_findings(
        self, chunks: List[RetrievalResult], error: str
    ) -> AnalystOutput:
        """Rule-based fallback — keyword classification when LLM fails."""
        findings: List[ComplianceFinding] = []
        for chunk in chunks:
            risk_level = self._classify_risk(chunk.chunk_text)
            findings.append(
                ComplianceFinding(
                    clause_reference=chunk.metadata.get("section", "Unknown"),
                    risk_level=risk_level,
                    finding=chunk.chunk_text[:100] + "...",
                    document_source=chunk.metadata.get("document", "Unknown"),
                )
            )
        return AnalystOutput(
            risk_findings=findings,
            result={
                "summary": f"Rule-based fallback: {len(findings)} finding(s)",
                "analyst_error": error[:200],
            },
            confidence=0.3,
            reasoning=f"LLM failed; rule-based classification applied. Error: {error[:100]}",
        )

    def _classify_risk(self, text: str) -> Literal["High", "Medium", "Low", "N/A"]:
        """Rule-based risk classification by keyword matching.

        Used as fallback when LLM is unavailable or returns invalid output.
        Returns one of the ComplianceFinding.risk_level Literal values.
        """
        text_lower = text.lower()

        if any(w in text_lower for w in ["violat", "non-compliance", "material weakness"]):
            return "High"
        if any(w in text_lower for w in ["requires", "must", "shall", "attention"]):
            return "Medium"
        if any(w in text_lower for w in ["optional", "recommended", "guidance"]):
            return "Low"
        return "N/A"
