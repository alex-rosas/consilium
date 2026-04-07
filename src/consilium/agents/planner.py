"""
PlannerAgent: Decomposes user queries into max 3 sub-tasks.

Phase 1: Makes the first real LLM call (Ollama, Llama 3.1 8B).
Phase 2: Switched to ChatOllama (chat interface) for reliable structured JSON output.
Falls back to a single-task plan if the LLM call fails.
"""

from __future__ import annotations

import json
import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from consilium.agents.base import BaseAgent
from consilium.schemas.agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

# Retry configuration — override in tests with wait_none() via monkeypatch
_LLM_RETRY_WAIT = wait_exponential(multiplier=0.5, min=0.5, max=4.0)


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
        from consilium.config import settings

        self._settings = settings

    @property
    def capabilities(self) -> List[str]:
        return ["task_planning", "query_decomposition"]

    async def execute(self, input: PlannerInput) -> PlannerOutput:
        """
        Decompose user query into sub-tasks.

        Calls ChatOllama with a system/user message pair for reliable JSON output.
        Creates a new LLM client per request for correct key rotation under concurrency.
        Falls back to single-task plan if LLM fails.
        """
        from consilium.integrations.llm_factory import create_llm_client

        llm = getattr(self, "llm", None) or create_llm_client(self._settings)
        sys_msg, user_msg = self._build_prompt(input.user_query)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=_LLM_RETRY_WAIT,
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    response = await llm.ainvoke([sys_msg, user_msg])
                    raw_text = response.content  # AIMessage.content is the string
                    task_plan = self._parse_llm_response(raw_text)

            return PlannerOutput(
                task_plan=task_plan,
                result={"plan_summary": self._summarize_plan(task_plan)},
                confidence=0.85,
                reasoning=f"LLM decomposed query into {len(task_plan)} task(s)",
            )

        except Exception as exc:
            logger.warning("PlannerAgent LLM call failed after retries, using fallback plan: %s", exc)
            return self._fallback_plan(input.user_query, str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, user_query: str) -> tuple[SystemMessage, HumanMessage]:
        """Return (SystemMessage, HumanMessage) for ChatOllama.ainvoke().

        System/user separation gives llama3.1:8b better role context and
        produces more reliable structured JSON output than a single string prompt.
        """
        system_content = (
            "You are a task planner for compliance automation.\n"
            "Output ONLY a valid JSON array — no markdown fences, no explanation.\n"
            "Schema: "
            '[{"task_id": "task_N", "description": "...", '
            '"assigned_agent": "analyst|synthesizer", "dependencies": []}]\n'
            "Rules:\n"
            "- Maximum 3 tasks\n"
            "- task_id format: task_1, task_2, task_3\n"
            "- Dependencies must reference task_ids in the same array\n"
            "- Available agents: analyst (risk analysis), synthesizer (report generation)\n"
            "- Most queries need 2 tasks: one analyst, one synthesizer"
        )
        user_content = f"Query: {user_query}"
        return SystemMessage(content=system_content), HumanMessage(content=user_content)

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
