# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync

# Copy the application code
COPY . .

# Expose the FastAPI application port
EXPOSE 8080

# Use uv venv to run the application
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]