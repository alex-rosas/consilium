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
from pydantic import ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from consilium.agents.base import BaseAgent
from consilium.integrations.retrieval_mock import RetrievalResult

# Prompt size guardrails — prevent LLM output truncation on large retrievals.
# Real Quaestor returns up to 20+ chunks for broad queries (e.g. JPMorgan filings).
# Feeding all chunks produces JSON responses of 10K+ tokens that get cut off mid-string.
# 6 chunks × 600 chars ≈ 3600 chars input → ~4 findings → ~600 token output.
_MAX_ANALYST_CHUNKS: int = 6
_MAX_CHUNK_CHARS: int = 600
from consilium.schemas.agent import AgentInput, AgentOutput
from consilium.schemas.findings import ComplianceFinding

logger = logging.getLogger(__name__)

# Retry configuration — override in tests with wait_none() via monkeypatch
_LLM_RETRY_WAIT = wait_exponential(multiplier=0.5, min=0.5, max=4.0)


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

        self._settings = settings

    @property
    def capabilities(self) -> List[str]:
        return ["risk_classification", "compliance_analysis", "regulatory_assessment"]

    async def execute(self, input: AnalystInput) -> AnalystOutput:
        """
        Classify compliance risks across all retrieved chunks in a single LLM call.

        Calls ChatOllama with chunk content and instructs it to output a JSON array
        of ComplianceFinding-compatible objects. Creates a new LLM client per request
        for correct key rotation under concurrency. Falls back to rule-based
        classification if the LLM fails or returns unparseable output.

        Args:
            input: AnalystInput with retrieved chunks.

        Returns:
            AnalystOutput with risk findings and confidence score.
        """
        from consilium.integrations.llm_factory import create_llm_client

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
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=_LLM_RETRY_WAIT,
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    # Create LLM client inside attempt so each retry rotates to the
                    # next Groq API key (round-robin in llm_factory). Without this,
                    # all retries hammer the same key and all fail on rate-limited keys.
                    llm = getattr(self, "llm", None) or create_llm_client(self._settings)
                    response = await llm.ainvoke([sys_msg, user_msg])
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
            logger.warning("AnalystAgent LLM failed after retries, using rule-based fallback: %s", exc)
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
            "Analyse the document chunks and identify the most critical compliance risks.\n"
            "Output ONLY a valid JSON array — no markdown fences, no explanation.\n"
            "Schema: "
            '[{"clause_reference": "...", "finding": "...", '
            '"risk_level": "High|Medium|Low|N/A", "document_source": "..."}]\n'
            "Rules:\n"
            "- clause_reference: specific regulation clause (e.g. 'IFRS 15.31')\n"
            "- finding: 20-100 chars MAXIMUM — one concise sentence only\n"
            "- risk_level: exactly one of High, Medium, Low, N/A\n"
            "- document_source: document identifier from metadata\n"
            "- Output MAXIMUM 3 findings — prioritise the highest-risk items\n"
            "- Do NOT repeat similar findings; pick the 3 most distinct risks\n"
            "- If no compliance risks found, output an empty array: []"
        )

        parts: List[str] = []

        if task_context:
            parts.append(f"Task plan:\n{task_context}")

        # Apply guardrails: cap chunk count and truncate long texts to keep the
        # total prompt within a range where llama-3.1-8b-instant can produce
        # complete, valid JSON without output truncation.
        capped_chunks = chunks[:_MAX_ANALYST_CHUNKS]
        if len(chunks) > _MAX_ANALYST_CHUNKS:
            logger.debug(
                "Capping analyst input from %d to %d chunks (guardrail)",
                len(chunks), _MAX_ANALYST_CHUNKS,
            )

        chunk_lines = []
        for i, chunk in enumerate(capped_chunks, 1):
            doc = chunk.metadata.get("document", "Unknown")
            section = chunk.metadata.get("section", "")
            label = f"{doc}" + (f" §{section}" if section else "")
            text = chunk.chunk_text[:_MAX_CHUNK_CHARS]
            if len(chunk.chunk_text) > _MAX_CHUNK_CHARS:
                text += "…"
            chunk_lines.append(f"Chunk {i} [{label}]:\n\"{text}\"")

        parts.append("Retrieved chunks:\n\n" + "\n\n".join(chunk_lines))
        user_content = "\n\n".join(parts)
        return SystemMessage(content=system_content), HumanMessage(content=user_content)

    def _parse_llm_response(self, response: str) -> List[ComplianceFinding]:
        """Parse and validate the LLM JSON response into ComplianceFinding objects.

        Strips markdown fences if present. Attempts partial recovery on truncated
        JSON before raising ValueError (which triggers the Tenacity retry).
        """
        text = response.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            inner = [ln for ln in lines[1:] if ln.strip() != "```"]
            text = "\n".join(inner).strip()

        # First try: parse as-is
        try:
            raw = json.loads(text)
            if not isinstance(raw, list):
                raise ValueError("Expected JSON array of finding objects")
            return [ComplianceFinding.model_validate(item) for item in raw]
        except json.JSONDecodeError:
            pass

        # Second try: recover truncated array by closing the last complete object.
        # Handles the case where Groq output is cut off mid-string due to token limits.
        try:
            start = text.find("[")
            if start == -1:
                raise ValueError("LLM returned no JSON array")
            # Find the last complete object boundary — last `},` or `}` before the end
            last_complete = max(text.rfind("},"), text.rfind("}\n"))
            if last_complete > start:
                repaired = text[start : last_complete + 1] + "]"
                raw = json.loads(repaired)
                if isinstance(raw, list) and raw:
                    logger.debug(
                        "Recovered %d finding(s) from truncated JSON (%d→%d chars)",
                        len(raw), len(text), len(repaired),
                    )
                    return [ComplianceFinding.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValueError):
            pass

        raise ValueError(f"LLM returned unparseable JSON (len={len(text)})")

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
