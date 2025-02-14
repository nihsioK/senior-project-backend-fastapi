# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install only the necessary runtime dependencies (faster installation)
RUN apt update && apt install -y --no-install-recommends \
    libavformat58 libavdevice58 libavcodec58 libavutil56 libswscale6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the FastAPI application port
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
