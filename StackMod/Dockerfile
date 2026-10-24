FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and group
RUN groupadd -r appuser && useradd -r -u 1000 -g appuser -d /app appuser

WORKDIR /app

# Copy package metadata and source code for installation from pyproject.toml
COPY pyproject.toml README.md ./
COPY core/ core/
COPY db/ db/
COPY service/ service/
COPY tools/ tools/

# Install from pyproject.toml and resolve opencv-python conflict by using opencv-python-headless
RUN pip install --no-cache-dir . \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless

# Create required directories and set proper ownership
RUN mkdir -p data/snapshots data/faiss_index models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
