# Use a Python image with `uv` pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Enable bytecode compilation for optimized performance
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy `pyproject.toml` and `uv.lock` first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using `uv sync`
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application code
COPY . /app

# Install the application itself separately
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# **Ensure `uvicorn` is installed and available**
RUN /app/.venv/bin/uv pip install uvicorn

# **Set the correct path to uvicorn**
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint
ENTRYPOINT []

# Run FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
