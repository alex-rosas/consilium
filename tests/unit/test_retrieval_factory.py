"""
Unit tests for the retrieval provider factory.

Verifies that create_retrieval_client() returns the correct client
based on the RETRIEVAL_PROVIDER environment variable.
"""

from __future__ import annotations

import pytest

from consilium.integrations.factory import create_retrieval_client
from consilium.integrations.quaestor_client import QuaestorClient
from consilium.integrations.retrieval_mock import MockRetrieval


class TestCreateRetrievalClient:
    def test_returns_mock_retrieval_by_default(self) -> None:
        """Default RETRIEVAL_PROVIDER=mock returns MockRetrieval."""
        from consilium.config import Settings

        settings = Settings(retrieval_provider="mock")
        client = create_retrieval_client(settings)
        assert isinstance(client, MockRetrieval)

    def test_returns_quaestor_client_when_configured(self) -> None:
        """RETRIEVAL_PROVIDER=quaestor returns QuaestorClient."""
        from consilium.config import Settings

        settings = Settings(retrieval_provider="quaestor")
        client = create_retrieval_client(settings)
        assert isinstance(client, QuaestorClient)

    def test_quaestor_client_uses_configured_base_url(self) -> None:
        """QuaestorClient is initialised with quaestor_base_url from settings."""
        from consilium.config import Settings

        settings = Settings(
            retrieval_provider="quaestor",
            quaestor_base_url="http://quaestor.internal:9000",
        )
        client = create_retrieval_client(settings)
        assert isinstance(client, QuaestorClient)
        assert client.base_url == "http://quaestor.internal:9000"

    def test_mock_retrieval_has_retrieve_method(self) -> None:
        """Both clients expose an async retrieve() method (same interface)."""
        from consilium.config import Settings
        import inspect

        settings = Settings(retrieval_provider="mock")
        client = create_retrieval_client(settings)
        assert hasattr(client, "retrieve")
        assert inspect.iscoroutinefunction(client.retrieve)

    def test_quaestor_client_has_retrieve_method(self) -> None:
        """QuaestorClient exposes the same async retrieve() interface."""
        from consilium.config import Settings
        import inspect

        settings = Settings(retrieval_provider="quaestor")
        client = create_retrieval_client(settings)
        assert hasattr(client, "retrieve")
        assert inspect.iscoroutinefunction(client.retrieve)
