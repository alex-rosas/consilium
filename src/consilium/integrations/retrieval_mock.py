"""
Mock retrieval interface for Phase 0-1 testing.
Simulates Quaestor's retrieval endpoint with deterministic data.

The RetrievalResult schema here MUST match Quaestor's API response shape.
This is the Phase 2 swap boundary — swap MockRetrieval for QuaestorClient
and the schema contract guarantees compatibility.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional


class RetrievalResult(BaseModel):
    """
    Single retrieval result.
    Schema must match Quaestor's /retrieve API response for Phase 2 swap.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_text: str = Field(..., description="Retrieved text chunk")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: float = Field(..., ge=0.0, description="Relevance score")


class MockRetrieval:
    """
    Deterministic mock for retrieval testing.

    Always returns the same 3 chunks regardless of query.
    This enables reproducible, fast agent testing without external dependencies.
    """

    FIXED_CHUNKS = [
        {
            "chunk_text": (
                "IFRS 15.31 requires an entity to recognize revenue when "
                "(or as) the entity satisfies a performance obligation by "
                "transferring a promised good or service to a customer."
            ),
            "metadata": {
                "document": "IFRS_15_Revenue_Recognition",
                "section": "31",
                "standard": "IFRS 15"
            },
            "score": 0.92
        },
        {
            "chunk_text": (
                "JPMorgan Chase reported Q3 2023 revenue of $39.9 billion, "
                "with significant growth in investment banking fees. Revenue "
                "recognition follows the five-step model under IFRS 15."
            ),
            "metadata": {
                "document": "JPM_10Q_2023Q3",
                "page": "42",
                "section": "Revenue Recognition"
            },
            "score": 0.88
        },
        {
            "chunk_text": (
                "Historical compliance findings indicate that revenue recognition "
                "practices in financial services require particular attention to "
                "contract identification and variable consideration estimates."
            ),
            "metadata": {
                "document": "PCAOB_Audit_Report_2023",
                "finding_id": "REV-2023-08",
                "risk_level": "Medium"
            },
            "score": 0.75
        }
    ]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[Dict[str, str]] = None
    ) -> List[RetrievalResult]:
        """
        Returns FIXED_CHUNKS regardless of query.

        Args:
            query: Search query (intentionally ignored in mock)
            top_k: Number of results to return
            document_filter: Optional filter (intentionally ignored in mock)

        Returns:
            Deterministic list of RetrievalResult objects
        """
        chunks = self.FIXED_CHUNKS[:top_k]
        return [RetrievalResult(**chunk) for chunk in chunks]
