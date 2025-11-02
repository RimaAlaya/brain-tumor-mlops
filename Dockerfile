# Root Dockerfile - Default serving image
# This is the main Dockerfile for quick deployment
# For specialized builds, use docker/Dockerfile.train or docker/Dockerfile.serve

FROM python:3.10-slim

LABEL maintainer="rima.alaya033@gmail.com"
LABEL description="Brain Tumor Classification API"
LABEL version="1.0.0"

WORKDIR /app

# Install system dependencies with pinned versions
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/brain_tumor_model.keras \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - API server (JSON notation)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]