# ==============================================================================
# Urban Traffic & Building Vision Appliance Dockerfile
# Base: Debian 12 (Bookworm) Slim + Python 3.11 + Intel AVX2 SIMD Vectorization
# ==============================================================================

FROM python:3.11-slim-bookworm AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# System libraries for OpenCV, PyAV, FFmpeg, and Zstandard
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    zstd \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and configurations
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create persistent storage mountpoints
RUN mkdir -p /data/media/merged/vision /data/media/merged/vision/validation /data/media/merged/vision/archives

EXPOSE 9099

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD curl -f http://127.0.0.1:9099/health || exit 1

# Default execution: run the unified API and vision worker
ENTRYPOINT ["python3", "-m", "src.api.server"]
