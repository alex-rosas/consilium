"""
PlannerAgent: Decomposes user queries into max 3 sub-tasks.

Phase 1: Makes the first real LLM call (Ollama, Llama 3.1 8B).
Falls back to a single-task plan if the LLM call fails.
"""

from __future__ import annotations

import json
import logging
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from consilium.agents.base import BaseAgent
from consilium.schemas.agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """Single sub-task in the Planner's decomposition."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Unique ID: task_1, task_2, or task_3")
    description: str = Field(..., min_length=10, max_length=500)
    assigned_agent: str = Field(..., description="Agent capability required (analyst|synthesizer)")
    dependencies: List[str] = Field(
        default_factory=list,
        description="task_ids that must complete before this task",
    )


class PlannerInput(AgentInput):
    """Input for the PlannerAgent."""

    model_config = ConfigDict(extra="forbid")

    user_query: str = Field(..., min_length=10, max_length=1000)


class PlannerOutput(AgentOutput):
    """Output: 1–3 validated sub-tasks."""

    model_config = ConfigDict(extra="forbid")

    task_plan: List[SubTask] = Field(..., min_length=1, max_length=3)

    @field_validator("task_plan")
    @classmethod
    def validate_dependencies(cls, v: List[SubTask]) -> List[SubTask]:
        """Reject dependency references to non-existent task_ids."""
        task_ids = {task.task_id for task in v}
        for task in v:
            invalid = set(task.dependencies) - task_ids
            if invalid:
                raise ValueError(f"Invalid dependencies: {invalid}")
        return v


# ---------------------------------------------------------------------------
# Agent implementation (Tasks 3+4 live in the same file per plan)
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    """
    Decomposes a user compliance query into 1–3 sub-tasks via Ollama LLM.

    On any LLM failure (connection error, malformed JSON, schema violation),
    returns a single-task fallback plan so the workflow can continue.
    """

    def __init__(self) -> None:
        from langchain_ollama import OllamaLLM

        from consilium.config import settings

        self.llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0,  # Deterministic for task planning
        )

    @property
    def capabilities(self) -> List[str]:
        return ["task_planning", "query_decomposition"]

    async def execute(self, input: PlannerInput) -> PlannerOutput:
        """
        Decompose user query into sub-tasks.

        Calls Ollama LLM with a structured JSON prompt.
        Falls back to single-task plan if LLM fails.
        """
        prompt = self._build_prompt(input.user_query)

        try:
            response = await self.llm.ainvoke(prompt)
            task_plan = self._parse_llm_response(response)
            return PlannerOutput(
                task_plan=task_plan,
                result={"plan_summary": self._summarize_plan(task_plan)},
                confidence=0.85,
                reasoning=f"LLM decomposed query into {len(task_plan)} task(s)",
            )

        except Exception as exc:
            logger.warning("PlannerAgent LLM call failed, using fallback plan: %s", exc)
            return self._fallback_plan(input.user_query, str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, user_query: str) -> str:
        return f"""You are a task planner for compliance automation.

User query: {user_query}

Decompose this query into 1-3 sub-tasks for a multi-agent system.

Available agent capabilities:
- "analyst": Retrieve and analyze compliance documents, classify risk
- "synthesizer": Aggregate findings into a structured report

Rules:
- Output ONLY valid JSON — no markdown fences, no explanation
- Maximum 3 tasks
- Each task: {{"task_id": "task_N", "description": "...", "assigned_agent": "analyst|synthesizer", "dependencies": []}}
- task_id format: "task_1", "task_2", "task_3"
- Dependencies must reference task_ids defined in the same array
- Most queries need 2 tasks: one analyst, one synthesizer

Output JSON array:"""

    def _parse_llm_response(self, response: str) -> List[SubTask]:
        """
        Parse and validate the LLM JSON response.

        Strips markdown fences if present, parses JSON, truncates to ≤3 tasks,
        then validates each task against the SubTask schema.

        Raises ValueError on any parse or validation failure.
        """
        text = response.strip()

        # Strip markdown code fences if the LLM added them despite instructions
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop opening fence (```json or ```) and closing fence (```)
            inner = [
                ln for ln in lines[1:] if ln.strip() != "```"
            ]
            text = "\n".join(inner).strip()

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError("Expected JSON array of task objects")

        # Silently truncate to max 3 — model sometimes ignores the instruction
        if len(raw) > 3:
            logger.debug("LLM returned %d tasks; truncating to 3", len(raw))
            raw = raw[:3]

        return [SubTask.model_validate(item) for item in raw]

    def _summarize_plan(self, tasks: List[SubTask]) -> str:
        return "; ".join(f"{t.task_id}: {t.description}" for t in tasks)

    def _fallback_plan(self, user_query: str, error: str) -> PlannerOutput:
        """Single-task fallback when LLM fails — analyst handles everything."""
        fallback = SubTask(
            task_id="task_1",
            description=f"Analyze compliance risks for: {user_query[:90]}",
            assigned_agent="analyst",
            dependencies=[],
        )
        return PlannerOutput(
            task_plan=[fallback],
            result={
                "plan_summary": "Fallback: single-task analyst execution",
                "planner_error": error[:200],
            },
            confidence=0.3,
            reasoning=f"LLM decomposition failed; single-task fallback applied. Error: {error[:100]}",
        )
