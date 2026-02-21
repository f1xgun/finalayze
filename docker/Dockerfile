FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application
COPY src/ src/
COPY config/ config/

# Run
CMD ["uv", "run", "uvicorn", "finalayze.main:app", "--host", "0.0.0.0", "--port", "8000"]
