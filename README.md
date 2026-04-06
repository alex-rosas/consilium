# Consilium

Multi-agent compliance automation system with LangGraph orchestration and strict schema contracts.

Built incrementally with phase-gated validation — each phase introduces ONE complexity and proves it works before proceeding.

## Current Phase

**Phase 3: Evaluation Harness & Baseline Measurement** — Confidence propagation, fallback observability, golden dataset, evaluation runner.

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v

# Start API server (requires Ollama running at localhost:11434)
uvicorn consilium.api.main:app --port 8001
```

## Running with Docker Compose

### Quick start (all platforms)

```bash
docker-compose up
```

The `extra_hosts` configuration automatically makes `host.docker.internal` work on all platforms — macOS, Windows, and Linux.

### Linux-specific notes

On native Linux Docker, `host.docker.internal` is not defined by default. The `extra_hosts: host.docker.internal:host-gateway` entry in `docker-compose.yml` adds it automatically.

If you still have connection issues, override the URLs manually:

```bash
# Find your Docker bridge IP
ip addr show docker0

# Override URLs (replace 172.17.0.1 with your actual bridge IP)
OLLAMA_BASE_URL=http://172.17.0.1:11434 \
QUAESTOR_BASE_URL=http://172.17.0.1:8000 \
docker-compose up
```

### Environment variable overrides

All Docker Compose settings can be overridden via environment variables or a `.env` file:

```bash
cp .env.docker.example .env
# edit .env with your values
docker-compose up
```

### Troubleshooting

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check Quaestor is running (if using quaestor retrieval)
curl http://localhost:8000/health

# Check Consilium API health
curl http://localhost:8001/health

# Inspect Docker network bridge
docker network inspect bridge
```

## Running Evaluation

```bash
# Requires API running on localhost:8001 and Ollama running
python eval/run_eval.py --phase phase2_baseline

# Custom API URL
python eval/run_eval.py --api-url http://localhost:8001 --phase my_run
```

Results are saved to `eval/results/` as timestamped JSON files.

## Stack

- **Python 3.11+** with `uv` package management
- **FastAPI** — Agent gateway API
- **Pydantic v2** — Strict schema validation on all agent I/O
- **LangGraph + LangChain** — Multi-agent orchestration (Phase 1+)
- **Ollama** — Local LLM inference (Phase 1+)
