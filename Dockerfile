# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock ./

# Ensure the virtual environment is created and install dependencies inside it
RUN uv venv .venv && .venv/bin/uv pip install --system

# Copy the application code
COPY . .

# Expose the FastAPI application port
EXPOSE 8080

# Use an absolute path to uvicorn
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
