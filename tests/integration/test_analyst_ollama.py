"""
Integration test: AnalystAgent with real ChatOllama.

Requires Ollama running at localhost:11434 with llama3.1:8b loaded.
Skip gracefully if Ollama is unavailable.

Gated with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest

from consilium.agents.analyst import AnalystAgent, AnalystInput


_CHUNKS = [
    {
        "chunk_text": (
            "IFRS 15.31 requires an entity to recognize revenue when (or as) "
            "the entity satisfies a performance obligation by transferring a "
            "promised good or service to a customer."
        ),
        "metadata": {"document": "IFRS_15", "section": "31"},
        "score": 0.92,
    },
    {
        "chunk_text": (
            "JPMorgan Chase reported Q3 2023 revenue of $39.9 billion. "
            "Revenue recognition follows the five-step model under IFRS 15."
        ),
        "metadata": {"document": "JPM_10Q_2023Q3", "section": "Revenue Recognition"},
        "score": 0.88,
    },
]


async def _ollama_is_available() -> bool:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
async def test_analyst_chatollama_produces_findings() -> None:
    """ChatOllama path returns ComplianceFinding objects — no fallback triggered."""
    if not await _ollama_is_available():
        pytest.skip("Ollama not running at localhost:11434")

    agent = AnalystAgent()
    output = await agent.execute(
        AnalystInput(task="Classify risks", context={}, retrieved_chunks=_CHUNKS)
    )

    # LLM path: confidence > fallback threshold (0.3)
    assert output.confidence > 0.5, (
        f"Analyst fell back to rule-based (confidence={output.confidence}). "
        f"Error: {output.result.get('analyst_error', 'unknown')}"
    )
    # Findings should be typed ComplianceFinding objects
    from consilium.schemas.findings import ComplianceFinding
    for f in output.risk_findings:
        assert isinstance(f, ComplianceFinding)
        assert f.risk_level in ("High", "Medium", "Low", "N/A")
