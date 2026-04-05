"""
Unit tests for QuaestorClient retry logic (Task 7).

Verifies:
- HTTP connection errors (RequestError) are retried up to 2 attempts
- HTTP status errors (4xx/5xx) are NOT retried
- After retry exhaustion, raises HTTPException(503)
"""

from __future__ import annotations

import pytest
import respx
import httpx
from tenacity import wait_none

from consilium.integrations.quaestor_client import QuaestorClient


QUAESTOR_URL = "http://quaestor-test:8000"
RETRIEVE_URL = f"{QUAESTOR_URL}/retrieve"

_VALID_RESPONSE = {
    "results": [
        {
            "chunk_text": "Revenue must be recognised when performance obligations are satisfied.",
            "score": 0.9,
            "metadata": {"document": "IFRS15", "section": "31"},
        }
    ]
}


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tenacity backoff so retry tests run instantly."""
    monkeypatch.setattr(
        "consilium.integrations.quaestor_client._QUAESTOR_RETRY_WAIT", wait_none()
    )


class TestQuaestorRetry:
    async def test_retries_on_request_error_then_succeeds(self) -> None:
        """RequestError on first attempt, success on second — 2 POST calls."""
        client = QuaestorClient(base_url=QUAESTOR_URL)
        call_count = 0

        async def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused", request=request)
            return httpx.Response(200, json=_VALID_RESPONSE)

        with respx.mock:
            respx.post(RETRIEVE_URL).mock(side_effect=_side_effect)
            results = await client.retrieve("IFRS 15 revenue recognition")

        assert call_count == 2
        assert len(results) == 1

    async def test_no_retry_on_http_status_error(self) -> None:
        """HTTPStatusError (404) is not retried — exactly 1 call, raises 503."""
        client = QuaestorClient(base_url=QUAESTOR_URL)

        with respx.mock:
            route = respx.post(RETRIEVE_URL).mock(
                return_value=httpx.Response(404, json={"detail": "Not found"})
            )
            with pytest.raises(Exception) as exc_info:
                await client.retrieve("IFRS 15 revenue recognition")

            # Only 1 call — HTTPStatusError must not be retried
            assert route.call_count == 1

        assert exc_info.value.status_code == 503  # type: ignore[attr-defined]

    async def test_exhausts_retries_raises_503(self) -> None:
        """RequestError on both attempts — raises HTTPException(503)."""
        client = QuaestorClient(base_url=QUAESTOR_URL)
        call_count = 0

        async def _always_fail(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Always fails", request=request)

        with respx.mock:
            respx.post(RETRIEVE_URL).mock(side_effect=_always_fail)
            with pytest.raises(Exception) as exc_info:
                await client.retrieve("IFRS 15 revenue recognition")

        assert call_count == 2  # 2 attempts total
        assert exc_info.value.status_code == 503  # type: ignore[attr-defined]
