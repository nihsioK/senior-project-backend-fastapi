# Use the official Python image
FROM python:3.12-slim AS builder

# Set the working directory
WORKDIR /app

# Install necessary system dependencies for aiortc and video decoding
RUN apt update && apt install -y --no-install-recommends \
    build-essential libffi-dev libssl-dev \
    libavformat-dev libavdevice-dev libavcodec-dev libavutil-dev libswscale-dev \
    ffmpeg gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly libx264-dev libx265-dev libvpx-dev libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Create the final lightweight runtime image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /venv /venv
COPY . .

# Expose the FastAPI application port
EXPOSE 8080

# Set environment variables for GStreamer and OpenCV
ENV GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="video_codec;h264"
ENV PATH="/venv/bin:$PATH"

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
