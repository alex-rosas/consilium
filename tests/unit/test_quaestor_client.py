"""
Unit tests for QuaestorClient.

All HTTP calls are mocked with respx — no real Quaestor required.
Tests cover: successful retrieval, HTTP errors, connection errors,
health check, and mock fallback behavior.
"""

from __future__ import annotations

import pytest
import httpx
import respx

from fastapi import HTTPException

from consilium.integrations.quaestor_client import QuaestorClient
from consilium.integrations.retrieval_mock import RetrievalResult


QUAESTOR_URL = "http://localhost:8000"

_VALID_RESULT = {
    "chunk_text": "IFRS 15.31 requires revenue recognition on performance obligation satisfaction.",
    "metadata": {"document": "IFRS_15", "section": "31"},
    "score": 0.95,
}


# ---------------------------------------------------------------------------
# Successful retrieval
# ---------------------------------------------------------------------------


class TestQuaestorClientRetrieve:
    @respx.mock
    async def test_successful_retrieval_returns_results(self) -> None:
        """Returns a list of RetrievalResult on HTTP 200."""
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(200, json={"results": [_VALID_RESULT]})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        results = await client.retrieve("IFRS 15 compliance", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score == 0.95
        assert results[0].chunk_text.startswith("IFRS 15.31")

    @respx.mock
    async def test_multiple_results_returned(self) -> None:
        """All results in the response are parsed."""
        chunks = [dict(_VALID_RESULT, score=0.9 - i * 0.1) for i in range(3)]
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(200, json={"results": chunks})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        results = await client.retrieve("test", top_k=3)

        assert len(results) == 3

    @respx.mock
    async def test_empty_results_list_accepted(self) -> None:
        """Empty results list is valid (no findings for query)."""
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        results = await client.retrieve("obscure query", top_k=5)

        assert results == []

    @respx.mock
    async def test_top_k_sent_in_request_body(self) -> None:
        """top_k parameter is forwarded in the POST body."""
        route = respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        await client.retrieve("test", top_k=7)

        sent_body = route.calls.last.request.content
        import json
        body = json.loads(sent_body)
        assert body["top_k"] == 7
        assert body["query"] == "test"

    @respx.mock
    async def test_trailing_slash_in_base_url_stripped(self) -> None:
        """Trailing slash in base_url does not double-slash the path."""
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        client = QuaestorClient(base_url=f"{QUAESTOR_URL}/")
        results = await client.retrieve("test")
        assert results == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestQuaestorClientErrors:
    @respx.mock
    async def test_http_500_raises_503(self) -> None:
        """HTTP 5xx from Quaestor is wrapped as HTTPException 503."""
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        with pytest.raises(HTTPException) as exc_info:
            await client.retrieve("test")
        assert exc_info.value.status_code == 503

    @respx.mock
    async def test_http_404_raises_503(self) -> None:
        """HTTP 4xx from Quaestor is also wrapped as HTTPException 503."""
        respx.post(f"{QUAESTOR_URL}/retrieve").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        with pytest.raises(HTTPException) as exc_info:
            await client.retrieve("test")
        assert exc_info.value.status_code == 503

    async def test_connection_refused_raises_503(self) -> None:
        """Connection refused (Quaestor not running) raises HTTPException 503."""
        client = QuaestorClient(base_url="http://localhost:19999")
        with pytest.raises(HTTPException) as exc_info:
            await client.retrieve("test")
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Mock fallback (ALLOW_MOCK_FALLBACK=true)
# ---------------------------------------------------------------------------


class TestQuaestorClientMockFallback:
    async def test_connection_error_falls_back_to_mock_when_flag_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ALLOW_MOCK_FALLBACK=true, connection errors return mock data."""
        monkeypatch.setenv("ALLOW_MOCK_FALLBACK", "true")

        # Re-import settings to pick up the env var change
        import importlib
        import consilium.config as cfg_module
        importlib.reload(cfg_module)
        # Patch the settings used inside quaestor_client
        import consilium.integrations.quaestor_client as qc_module
        importlib.reload(qc_module)
        from consilium.integrations.quaestor_client import QuaestorClient as ReloadedClient

        client = ReloadedClient(base_url="http://localhost:19999")
        results = await client.retrieve("test", top_k=3)

        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)

        # Restore
        monkeypatch.delenv("ALLOW_MOCK_FALLBACK", raising=False)
        importlib.reload(cfg_module)
        importlib.reload(qc_module)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestQuaestorClientHealthCheck:
    @respx.mock
    async def test_health_check_true_on_200(self) -> None:
        """health_check() returns True when Quaestor responds 200."""
        respx.get(f"{QUAESTOR_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        assert await client.health_check() is True

    @respx.mock
    async def test_health_check_false_on_500(self) -> None:
        """health_check() returns False when Quaestor responds with error."""
        respx.get(f"{QUAESTOR_URL}/health").mock(
            return_value=httpx.Response(500)
        )
        client = QuaestorClient(base_url=QUAESTOR_URL)
        assert await client.health_check() is False

    async def test_health_check_false_when_unreachable(self) -> None:
        """health_check() returns False (never raises) when unreachable."""
        client = QuaestorClient(base_url="http://localhost:19999")
        result = await client.health_check()
        assert result is False
