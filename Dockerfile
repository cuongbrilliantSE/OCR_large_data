FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Install system dependencies:
# - p7zip-full for high-speed archive decompression (.rar, .zip, .7z)
# - libgl1 & libglib2.0-0 for headless OpenCV image processing
# - curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    p7zip-full \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . $HOME/app

# Ensure storage directories exist with write access
RUN mkdir -p data/input data/output logs && \
    chmod -R 755 /home/user/app/bin/* 2>/dev/null || true

EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/books || exit 1

# Start FastAPI Web application on port 7860 (Hugging Face default)
CMD ["python", "main.py", "web", "--host", "0.0.0.0", "--port", "7860"]

