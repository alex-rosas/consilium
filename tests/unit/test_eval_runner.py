"""Unit tests for eval runner trace_id extraction."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from run_eval import CaseResult, GoldenCase, run_case


def _make_case() -> GoldenCase:
    return GoldenCase(
        id="test-001",
        query="What are the IFRS 15 revenue recognition requirements?",
        expected_risk_levels=["High"],
        expected_agents_invoked=["planner", "analyst", "synthesizer"],
        expected_min_findings=1,
        requires_retrieval=False,
        label="test",
    )


@pytest.mark.asyncio
async def test_run_case_captures_trace_id_on_success() -> None:
    """run_case extracts trace_id from successful API response."""
    case = _make_case()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "confidence": 0.85,
        "trace_id": "abc123def456abc123def456abc12345",
        "result": {
            "risk_findings": [
                {
                    "clause_reference": "IFRS 15.31",
                    "risk_level": "High",
                    "finding": "Revenue recognition timing issue found here",
                    "document_source": "test",
                }
            ],
            "summary": "Risk identified" * 10,
        },
        "agents_invoked": ["planner", "analyst", "synthesizer"],
        "fallback_events": [],
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await run_case(mock_client, "http://test", case, True, store_responses=False)

    assert result.trace_id == "abc123def456abc123def456abc12345"


@pytest.mark.asyncio
async def test_run_case_trace_id_none_on_exception() -> None:
    """run_case sets trace_id to None when an exception occurs."""
    case = _make_case()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))

    result = await run_case(mock_client, "http://test", case, True, store_responses=False)

    assert result.trace_id is None
    assert result.success is False
