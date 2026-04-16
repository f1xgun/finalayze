FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and source (uv sync needs the package to exist)
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY config/ config/

# Install dependencies
RUN uv sync --frozen --no-dev

# Run
CMD ["uv", "run", "uvicorn", "finalayze.main:app", "--host", "0.0.0.0", "--port", "8000"]
