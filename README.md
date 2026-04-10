# Consilium — Multi-Agent Compliance Automation

A three-agent compliance analysis system built with LangGraph, FastAPI, and real-time SSE streaming. Submit a compliance query and watch the Planner, Analyst, and Synthesizer agents execute in your browser as the pipeline runs.

---

## Demo

Visit `http://localhost:8001` after starting the stack. Type a query like:

> _"What are the IFRS 15 revenue recognition risks in these quarterly filings?"_

The browser streams the pipeline in real time:

```
Planner     → 3 tasks planned
Analyst     → finding 1/3: IFRS 15.31 — High risk
              finding 2/3: IFRS 9.5.1 — Medium risk
              finding 3/3: IAS 1.15   — Low risk
Synthesizer → ## Executive Summary ...
Done        → View trace in Phoenix →
```

---

## What It Does

- **Decomposes** any natural-language compliance query into structured analysis tasks (PlannerAgent, LLM-driven)
- **Retrieves and classifies** relevant regulatory clauses against provided documents, producing risk-rated findings per clause (AnalystAgent + Quaestor retrieval)
- **Synthesizes** a plain-English compliance report from all findings (SynthesizerAgent), streamed to the browser via SSE as each agent completes

---

## Architecture

```
Browser
  │  GET /                    → streaming UI (single HTML file, no build step)
  │  POST /workflow/stream    → SSE event stream (fetch + ReadableStream)
  ▼
FastAPI (port 8001)
  │
  ▼
LangGraph (directed graph, async)
  ├── PlannerAgent     → LLM call → task_plan[], planner_confidence
  ├── AnalystAgent     → Quaestor /retrieve → LLM call → risk_findings[]
  └── SynthesizerAgent → LLM call → final_report
        │                    │
        ▼                    ▼
    Groq LLM            Quaestor API (port 8000)
    (llama-3.1-8b)      document retrieval service

        │
        ▼ OTLP HTTP
    Arize Phoenix (port 6006)
    per-agent OTel spans, trace viewer UI
```

SSE event types emitted as each node completes:

| Event               | Payload                                      |
|---------------------|----------------------------------------------|
| `planner_complete`  | `task_count`, `confidence`                   |
| `analyst_finding`   | `finding` (clause, risk_level, text)         |
| `analyst_complete`  | `findings_count`, `confidence`               |
| `report_complete`   | `report` (full markdown text)                |
| `done`              | `trace_id` (Phoenix link)                    |

---

## How to Run

### Quick Start (Docker)

Requires both repos co-located as siblings:

```
projects/
  consilium/   ← this repo
  quaestor/    ← document retrieval service (sibling directory)
```

Copy `.env.example` to `.env` and set your Groq key:

```bash
cp .env.example .env
# Edit .env — set at minimum: GROQ_API_KEY_1=gsk_...
```

Start the full stack with a single command:

```bash
docker-compose --profile app up
```

| Service   | Port | URL                   |
|-----------|------|-----------------------|
| Consilium | 8001 | http://localhost:8001 |
| Quaestor  | 8000 | http://localhost:8000 |
| Phoenix   | 6006 | http://localhost:6006 |

To start Phoenix alone (observability without the full app):

```bash
docker-compose --profile monitoring up
```

### Development Setup

```bash
# Install uv, then:
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# Required: GROQ_API_KEY_1
# Optional: GROQ_API_KEY_2, GROQ_API_KEY_3  (round-robin rotation)
#           RETRIEVAL_PROVIDER=quaestor       (default: mock)
#           QUAESTOR_BASE_URL=http://localhost:8000
#           OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
#           CONFIDENCE_THRESHOLD=0.5
#           TRACE_SAMPLE_RATE=1.0

# Optional: start Phoenix
docker-compose --profile monitoring up -d

# Optional: start Quaestor (from sibling directory)
cd ../quaestor && uvicorn quaestor.api.main:app --port 8000

# Start Consilium
uvicorn consilium.api.main:app --port 8001
```

**Run tests:**

```bash
.venv/bin/pytest tests/unit/ -v
```

**Run eval:**

```bash
.venv/bin/python eval/run_eval.py --phase phase7 --store-responses
```

### Environment Variables

| Variable                        | Default                   | Description                                    |
|---------------------------------|---------------------------|------------------------------------------------|
| `GROQ_API_KEY_1`                | _(required)_              | Primary Groq API key                           |
| `GROQ_API_KEY_2`                | —                         | Optional second key (round-robin rotation)     |
| `GROQ_API_KEY_3`                | —                         | Optional third key                             |
| `LLM_PROVIDER`                  | `groq`                    | LLM backend (`groq` or `ollama`)               |
| `RETRIEVAL_PROVIDER`            | `mock`                    | `mock` or `quaestor`                           |
| `QUAESTOR_BASE_URL`             | `http://localhost:8000`   | Quaestor service endpoint                      |
| `OTEL_EXPORTER_OTLP_ENDPOINT`  | `http://localhost:4318`   | Phoenix OTLP HTTP endpoint                     |
| `CONFIDENCE_THRESHOLD`          | `0.5`                     | Minimum confidence to accept agent output      |
| `TRACE_SAMPLE_RATE`             | `1.0`                     | OTel trace sampling rate (0.0–1.0)             |

---

## Key Engineering Decisions

### 1. LangGraph for agent orchestration (not raw async chaining)

The three agents could have been chained as plain `async` function calls. LangGraph was chosen because it makes the graph topology explicit and inspectable, enforces a single shared `WorkflowState` TypedDict as the data contract between agents, and provides per-node streaming via `astream()` — which is how SSE events are generated without any additional plumbing. The graph structure also made it straightforward to attach per-node OTel spans. The cost is a dependency on LangGraph's API surface; the benefit is a pipeline that is testable at the node level in complete isolation.

### 2. Groq + `llama-3.1-8b-instant` (not Ollama for eval)

Early phases used Ollama for fully local inference. Groq was introduced in Phase 6 for its sub-second inference latency and free API tier, which made running a 30-case automated eval practical without a local GPU. Up to three API keys are rotated round-robin in a thread-safe manner so that sequential eval runs hit the rate limit more slowly. The model (`llama-3.1-8b-instant`) balances speed against reasoning quality for structured JSON extraction tasks — the system's primary LLM workload.

### 3. SSE streaming (not polling or WebSockets)

The browser needs to display agent output as it arrives. Server-Sent Events over a single HTTP connection are the simplest solution: one `POST /workflow/stream` request, one persistent response body, no additional state. WebSockets would add bidirectional complexity that is not needed; polling would require a job queue and separate state storage. The one constraint is that `EventSource` (the browser's native SSE API) does not support `POST` — so the UI uses `fetch` with `ReadableStream` instead, which is equally simple and handles backpressure correctly.

### 4. Pydantic `extra="forbid"` on all models

Every schema — `WorkflowRequest`, `WorkflowResponse`, `ComplianceFinding`, all SSE event types — carries `model_config = ConfigDict(extra="forbid")`. This means any LLM output or API payload that contains unexpected fields raises a `ValidationError` at the boundary rather than silently passing through. In a compliance context where findings are authoritative outputs, silent data loss is worse than a loud failure. The constraint also forces every agent prompt to be precise about the JSON schema it expects back from the LLM.

### 5. Phase-gated development with numeric eval gates

Each phase had a defined exit criterion (test count, eval pass rate, or specific capability). This prevented scope creep within phases and created a commit history that mirrors the architecture's growth. The eval gate (≥65% workflow completion, P95≤3s) was not met in Phase 6 (76.7%) due to the analyst agent sending uncapped LLM prompts from large Quaestor retrievals — JSON output truncated mid-string on Groq's free tier. Phase 7 fixed this with prompt guardrails (`_MAX_ANALYST_CHUNKS=6`) and per-retry API key rotation, achieving **100% completion (30/30)** with P95=2895ms. The gate structure served its purpose: it surfaced a real architectural issue rather than masking it.

---

## What Was Intentionally Not Built

| Item | Rationale |
|------|-----------|
| **Neo4j graph memory** | The Phase 3 bottleneck gate (cross-document failure rate ≥ 30%) was never triggered. Adding graph memory without a demonstrated need would be complexity for its own sake. |
| **LoRA fine-tuning** | Base model accuracy is adequate for structured extraction at this scale. Fine-tuning would require a labelled dataset, a GPU, and a serving pipeline — none of which improve the portfolio signal relative to the cost. |
| **Auth / API keys** | Consilium is a local development tool. Adding authentication addresses no real threat model for this deployment context and would obscure the compliance logic. |
| **Paid Groq tier** | Free-tier works with guardrailed prompts (`_MAX_ANALYST_CHUNKS=6`). Upgrading the tier would allow removing this cap for richer findings, but is an operational decision rather than an architectural one. |
| **Shared Quaestor/Consilium schema package** | `RetrievalResult` is independently defined in both projects. Extracting it into a third shared package would require publishing infrastructure and versioning. Duplication-by-convention is acceptable at this scale. |
| **OTel Collector** | Direct export to Phoenix via OTLP HTTP is sufficient. An OTel Collector adds another container and configuration surface with no observability benefit at single-service scale. |
| **Quaestor retrieval as a separate OTel span** | Retrieval duration is captured inside the analyst agent span. A nested child span would add instrumentation complexity for marginal observability gain. |

---

## Evaluation Results

| Phase | Description | Unit Tests | Eval Cases | Completion |
|-------|-------------|-----------|------------|------------|
| 0 | Single-agent baseline (Analyst only, mock retrieval) | 18 | 5/5 | 100% |
| 1 | LangGraph 3-agent pipeline (mock retrieval) | 62 | 10/10 | 100% |
| 2 | Quaestor integration (real retrieval) | 89 | 10/10 | 100% |
| 3 | Guardrails + retry logic | 120 | 10/10 | 100% |
| 4 | Confidence threshold + task context propagation | 148 | 15/15 | 100% |
| 5 | OTel tracing + Phoenix | 172 | 20/20 | 100% |
| 6 | Per-request LLM key rotation + SSE streaming | 202 | 30/30 | 76.7% |
| **7** | **Streaming web UI + Docker stack + analyst guardrails** | **222** | **30/30** | **100%** |

---

## Known Technical Debt

1. **SSE error path** — Backend exceptions drop the connection without emitting an `error` event. The UI handles dropped streams gracefully.
2. **Quaestor retrieval not in its own OTel span** — Captured inside the analyst agent span but not as a named child span.
3. **`RetrievalResult` schema duplication** — Defined independently in both Consilium and Quaestor.
4. **Single-threaded LLM calls** — Each agent's LLM `execute()` call is awaited in async context but blocks the event loop for its duration. True parallelism requires `asyncio.to_thread` offload.
5. **Analyst output caps finding count to 3** — `_MAX_ANALYST_CHUNKS=6` and max-3-findings prompt ensure JSON completeness on free-tier Groq. A production deployment with higher token limits could remove this cap for richer outputs.

---

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.104+ |
| Orchestration | LangGraph + LangChain |
| LLM | Groq (`llama-3.1-8b-instant`), round-robin key rotation |
| Retrieval | Quaestor (factory-switchable; mock default) |
| Schema validation | Pydantic v2, `extra="forbid"` everywhere |
| Tracing | OpenTelemetry → Arize Phoenix |
| Runtime | Python 3.11, uv |
| Packaging | Docker Compose (`--profile app`) |
