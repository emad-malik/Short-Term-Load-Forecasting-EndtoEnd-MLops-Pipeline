# Use Python 3.11 slim image (3.14 not yet available in Docker, 3.11 is stable and compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=80

# PORT=80 above is only the local/default fallback (used by `docker run` and
# docker-compose). Render injects its own PORT env var at runtime (typically
# 10000) which overrides this default automatically — the app must bind to
# whatever $PORT resolves to, not a hardcoded port. See DEPLOYMENT_RENDER.md.

# Install system dependencies (curl is needed for the HEALTHCHECK below)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . /app

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/app/static

# Expose port 80 for local/documentation purposes only.
# Render does not read EXPOSE — it routes traffic to whatever port the
# container binds to via the injected $PORT env var.
EXPOSE 80

# Health check (shell form so $PORT is expanded at runtime).
# Note: Render does not use this Docker HEALTHCHECK instruction for routing
# decisions — it uses the `healthCheckPath` set in render.yaml against the
# same $PORT. This instruction still matters for local `docker run` /
# docker-compose and for any other container orchestrator.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application, binding to Render's dynamically assigned port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
