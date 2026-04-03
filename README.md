# Consilium

Multi-agent compliance automation system with LangGraph orchestration and strict schema contracts.

Built incrementally with phase-gated validation — each phase introduces ONE complexity and proves it works before proceeding.

## Current Phase

**Phase 0: Foundation & Contracts** — Strict schemas, mock retrieval, single agent.

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Run tests
pytest tests/unit/ -v

# Start API server
uvicorn consilium.api.main:app --reload --port 8001
```

## Stack

- **Python 3.11+** with `uv` package management
- **FastAPI** — Agent gateway API
- **Pydantic v2** — Strict schema validation on all agent I/O
- **LangGraph + LangChain** — Multi-agent orchestration (Phase 1+)
- **Ollama** — Local LLM inference (Phase 1+)
