"""
Retrieval provider factory.

Returns the correct retrieval client based on settings.retrieval_provider:
  "mock"     → MockRetrieval (deterministic, no external dependency)
  "quaestor" → QuaestorClient (real HTTP calls to Quaestor service)

Both clients expose the same async retrieve() interface so the caller
(api/main.py) treats them identically.
"""

from __future__ import annotations

from typing import Union

from consilium.config import Settings
from consilium.integrations.quaestor_client import QuaestorClient
from consilium.integrations.retrieval_mock import MockRetrieval


def create_retrieval_client(settings: Settings) -> Union[MockRetrieval, QuaestorClient]:
    """
    Instantiate and return the configured retrieval client.

    Args:
        settings: Application settings (reads retrieval_provider + quaestor_base_url).

    Returns:
        MockRetrieval when retrieval_provider == "mock".
        QuaestorClient(quaestor_base_url) when retrieval_provider == "quaestor".
    """
    if settings.retrieval_provider == "quaestor":
        return QuaestorClient(base_url=settings.quaestor_base_url)
    return MockRetrieval()
