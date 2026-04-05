"""
QuaestorClient: async HTTP client for Quaestor's retrieval API.

Phase 2: Replaces MockRetrieval in the workflow when RETRIEVAL_PROVIDER=quaestor.
Quaestor runs as an independent service (separate repo, separate process).
Consilium never imports from the Quaestor Python package — HTTP only.

Error handling philosophy:
- Production (ALLOW_MOCK_FALLBACK=false): any failure raises HTTPException 503.
  Never silently serve stale or mock data.
- Dev/test (ALLOW_MOCK_FALLBACK=true): connection errors fall back to MockRetrieval.
  Explicit opt-in only — never the default.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx
from fastapi import HTTPException

from consilium.integrations.retrieval_mock import MockRetrieval, RetrievalResult

logger = logging.getLogger(__name__)


class QuaestorClient:
    """
    Async HTTP client for Quaestor's /retrieve endpoint.

    Implements the same interface as MockRetrieval so the two are interchangeable
    via dependency injection. Uses httpx.AsyncClient internally.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """
        Initialise the client.

        Args:
            base_url: Base URL of the Quaestor service (no trailing slash needed).
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=10.0)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[dict[str, str]] = None,
    ) -> List[RetrievalResult]:
        """
        Call Quaestor's POST /retrieve endpoint.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            document_filter: Optional metadata filters (passed through to Quaestor).

        Returns:
            List of RetrievalResult objects parsed from the response.

        Raises:
            HTTPException(503): When Quaestor is unreachable or returns an error,
                unless ALLOW_MOCK_FALLBACK=true (dev mode).
        """
        from consilium.config import settings

        payload: dict[str, object] = {"query": query, "top_k": top_k}
        if document_filter:
            payload["document_filter"] = document_filter

        try:
            response = await self._client.post(
                f"{self.base_url}/retrieve",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return [RetrievalResult.model_validate(r) for r in data.get("results", [])]

        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if settings.allow_mock_fallback:
                logger.warning(
                    "Quaestor unavailable (%s) — falling back to MockRetrieval (dev mode)",
                    exc,
                )
                return await MockRetrieval().retrieve(query, top_k)
            raise HTTPException(
                status_code=503,
                detail=f"Quaestor service unavailable: {exc}",
            ) from exc

    async def health_check(self) -> bool:
        """
        Check whether Quaestor is reachable and healthy.

        Returns:
            True if Quaestor responds with HTTP 200, False otherwise.
            Never raises — designed for startup checks and /health endpoints.
        """
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception:
            return False
