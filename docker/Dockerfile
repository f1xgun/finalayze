FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies.
# BuildKit cache mount keeps uv's wheel cache between builds — see Dockerfile.prod
# for the full rationale.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev

# Copy application
COPY src/ src/
COPY config/ config/

# Run
CMD ["uv", "run", "uvicorn", "finalayze.main:app", "--host", "0.0.0.0", "--port", "8000"]
