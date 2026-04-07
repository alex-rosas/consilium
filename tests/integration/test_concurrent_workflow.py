"""
Integration test: five concurrent workflow requests must all complete correctly.

Requires a running API server:
    .venv/bin/uvicorn consilium.api.main:app --port 8001

Run with:
    .venv/bin/pytest tests/integration/test_concurrent_workflow.py -v -m integration

This test is intentionally excluded from the default unit test run.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

_BASE_URL = "http://localhost:8001"
_TIMEOUT = 120.0  # Groq rate-limit retries can be slow

_WORKFLOW_PAYLOAD = {
    "query": "Assess the compliance risk of JPMorgan Q3 2023 revenue recognition against IFRS 15",
    "document_chunks": [],
}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_five_concurrent_workflows() -> None:
    """Five simultaneous requests must all complete without state corruption."""
    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        tasks = [
            client.post("/workflow", json=_WORKFLOW_PAYLOAD) for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses), (
        f"Non-200 responses: {[(r.status_code, r.text[:200]) for r in responses if r.status_code != 200]}"
    )

    confidences = [r.json()["confidence"] for r in responses]
    assert all(isinstance(c, float) for c in confidences)
    assert all(c >= 0.0 for c in confidences)

    # Each request must have a unique trace_id (no shared state between requests)
    trace_ids = [r.json().get("trace_id") for r in responses]
    non_null_ids = [t for t in trace_ids if t is not None]
    if non_null_ids:
        assert len(set(non_null_ids)) == len(non_null_ids), (
            f"Duplicate trace_ids detected — possible state sharing: {trace_ids}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_produce_independent_reports() -> None:
    """Reports from concurrent requests must be non-empty and structurally valid."""
    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=_TIMEOUT) as client:
        tasks = [
            client.post("/workflow", json=_WORKFLOW_PAYLOAD) for _ in range(3)
        ]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses)

    for resp in responses:
        body = resp.json()
        assert "status" in body
        assert "confidence" in body
        assert body["confidence"] >= 0.0
