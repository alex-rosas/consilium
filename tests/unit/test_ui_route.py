"""
Unit tests for the streaming web UI route.

GET / must return 200 with an HTML response containing the single-page UI.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture()
def test_client():
    """Return an ASGI test transport wrapping the FastAPI app."""
    from consilium.api.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class TestUIRoute:
    async def test_get_root_returns_200(self, test_client: httpx.AsyncClient) -> None:
        """GET / must return HTTP 200."""
        async with test_client as client:
            response = await client.get("/")
        assert response.status_code == 200

    async def test_get_root_content_type_is_html(self, test_client: httpx.AsyncClient) -> None:
        """GET / must return Content-Type: text/html."""
        async with test_client as client:
            response = await client.get("/")
        assert "text/html" in response.headers["content-type"]

    async def test_get_root_contains_html_tag(self, test_client: httpx.AsyncClient) -> None:
        """GET / response body must contain an <html> element."""
        async with test_client as client:
            response = await client.get("/")
        assert "<html" in response.text.lower()

    async def test_get_root_contains_form(self, test_client: httpx.AsyncClient) -> None:
        """GET / response body must contain a form for query submission."""
        async with test_client as client:
            response = await client.get("/")
        assert "<form" in response.text.lower()

    async def test_get_root_references_stream_endpoint(self, test_client: httpx.AsyncClient) -> None:
        """GET / JS must reference the /workflow/stream endpoint."""
        async with test_client as client:
            response = await client.get("/")
        assert "/workflow/stream" in response.text
