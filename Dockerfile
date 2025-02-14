# Use an official Python runtime as a base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install uv package manager globally
#RUN pip install uv

# Copy the pyproject.toml and uv.lock files first (to leverage caching)
COPY pyproject.toml uv.lock ./

# Ensure a virtual environment is created and install dependencies
RUN uv venv .venv && .venv/bin/uv pip install --system

# Copy the rest of the application code
COPY . .

# Expose the FastAPI application port
EXPOSE 8080

# **DEBUG STEP** Ensure everything is installed correctly
RUN ls -la /app/.venv/bin/

# Use an absolute path to uvicorn
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
