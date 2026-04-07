# ============================================================
# OpenEnv Data Cleaning Environment — Dockerfile
# ============================================================
# Base image: python:3.10-slim
# Port:       7860 (HuggingFace Spaces default)
# Entrypoint: uvicorn server:app
# ============================================================

FROM python:3.10-slim

# ---- Metadata ----
LABEL maintainer="OpenEnv Hackathon Submission"
LABEL description="Data Cleaning & Validation OpenEnv environment"
LABEL version="1.0.0"

# ---- System dependencies ----
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Python dependencies ----
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt


# ---- Application code ----
COPY . .

# ---- Environment variables ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    LOG_LEVEL=info

# ---- Expose HuggingFace Spaces port ----
EXPOSE 7860

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ---- Run server ----
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
