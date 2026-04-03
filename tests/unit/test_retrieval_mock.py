"""
Unit tests for mock retrieval interface.
Validates deterministic behavior and schema compliance.
"""

import pytest
from consilium.integrations.retrieval_mock import MockRetrieval, RetrievalResult


@pytest.mark.asyncio
class TestMockRetrieval:

    async def test_returns_same_chunks_regardless_of_query(self) -> None:
        mock = MockRetrieval()
        result1 = await mock.retrieve("query one", top_k=3)
        result2 = await mock.retrieve("completely different query", top_k=3)
        assert len(result1) == len(result2) == 3
        assert result1[0].chunk_text == result2[0].chunk_text

    async def test_respects_top_k(self) -> None:
        mock = MockRetrieval()
        result = await mock.retrieve("test", top_k=2)
        assert len(result) == 2

    async def test_top_k_larger_than_chunks_returns_all(self) -> None:
        mock = MockRetrieval()
        result = await mock.retrieve("test", top_k=100)
        assert len(result) == len(MockRetrieval.FIXED_CHUNKS)

    async def test_all_results_are_valid_schema(self) -> None:
        mock = MockRetrieval()
        results = await mock.retrieve("test", top_k=3)
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert 0.0 <= result.score <= 1.0
            assert result.chunk_text != ""
            assert isinstance(result.metadata, dict)
