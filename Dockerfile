# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies for aiortc and video decoding
RUN apt update && apt install -y --no-install-recommends \
    libavformat-dev libavdevice-dev libavcodec-dev libavutil-dev libswscale-dev \
    ffmpeg gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly libx264-dev libx265-dev libvpx-dev libopus-dev \
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
