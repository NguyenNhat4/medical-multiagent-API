# ============================================
# Stage 1: Build dependencies and models
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Set environment variables for better download stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    REQUESTS_TIMEOUT=300 \
    PIP_DEFAULT_TIMEOUT=300

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        wget \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --retries 5 --timeout 300 -r requirements.txt

# Create cache directory for models
RUN mkdir -p /app/models

# Preload models with retry logic (this layer will be cached unless preload_models.py changes)
COPY preload_models.py .
RUN python preload_models.py && rm preload_models.py

# ============================================
# Stage 2: Runtime image
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FASTEMBED_CACHE_PATH=/app/models

# Install only runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy preloaded models from builder
COPY --from=builder /app/models /app/models

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/logs \
    && chown -R app:app /app \
    && chmod -R 755 /app/logs \
    && chmod -R 755 /app/models
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command to run the application
CMD ["python", "start_api.py"]
