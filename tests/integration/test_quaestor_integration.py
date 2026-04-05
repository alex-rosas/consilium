"""
Integration test: QuaestorClient with real Quaestor service.

Requires Quaestor running at localhost:8000.
Skip gracefully if unavailable.

Gated with @pytest.mark.integration — not run in standard unit test suite.
"""

from __future__ import annotations

import pytest

from consilium.integrations.quaestor_client import QuaestorClient
from consilium.integrations.retrieval_mock import RetrievalResult


async def _quaestor_is_available(client: QuaestorClient) -> bool:
    """Return True if Quaestor health check passes."""
    return await client.health_check()


@pytest.mark.integration
async def test_quaestor_health_check() -> None:
    """QuaestorClient.health_check() returns True when Quaestor is running."""
    client = QuaestorClient("http://localhost:8000")
    if not await _quaestor_is_available(client):
        pytest.skip("Quaestor not running at localhost:8000")
    assert await client.health_check() is True


@pytest.mark.integration
async def test_quaestor_retrieve_returns_results() -> None:
    """QuaestorClient.retrieve() returns at least one RetrievalResult."""
    client = QuaestorClient("http://localhost:8000")
    if not await _quaestor_is_available(client):
        pytest.skip("Quaestor not running at localhost:8000")

    results = await client.retrieve(
        "revenue recognition IFRS 15 JPMorgan",
        top_k=5,
    )

    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(len(r.chunk_text) > 0 for r in results)
