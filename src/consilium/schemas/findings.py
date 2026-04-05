"""
ComplianceFinding — unified domain model shared between AnalystAgent and SynthesizerAgent.

Phase 2: Replaces the fragile AnalystAgent→SynthesizerAgent field mapping that lived
in WorkflowGraph._synthesizer_node(). Both agents now use this single schema so the
graph can pass findings through without any translation.

Field rationale:
- clause_reference: It's a reference to a clause, not the clause text itself
- finding: Encompasses both evidence and analytical interpretation
- risk_level: Literal enum — no free-form strings, validation at boundary
- document_source: Clearer than "document" — attribution intent is explicit
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ComplianceFinding(BaseModel):
    """
    Single compliance risk finding.

    Shared between AnalystAgent output (List[ComplianceFinding]) and
    SynthesizerAgent input (List[ComplianceFinding]). Using this unified
    schema eliminates the field mapping layer that existed in Phase 1.
    """

    model_config = ConfigDict(extra="forbid")

    clause_reference: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Regulatory clause reference (e.g., 'IFRS 15.31', 'PCAOB AS 2810')",
    )
    finding: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Evidence and analytical interpretation of the compliance issue",
    )
    risk_level: Literal["High", "Medium", "Low", "N/A"] = Field(
        ...,
        description="Risk classification: High | Medium | Low | N/A",
    )
    document_source: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Source document identifier (e.g., 'JPMorgan 10-K Q3 2023')",
    )
