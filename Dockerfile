FROM python:3.11-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer cache)
COPY pyproject.toml ./
COPY src/ ./src/

# Install production dependencies only (no dev extras)
RUN uv pip install --system --no-cache .

# Download spaCy model required by InputGuardrails
RUN python -m spacy download en_core_web_lg

EXPOSE 8001

CMD ["uvicorn", "consilium.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
