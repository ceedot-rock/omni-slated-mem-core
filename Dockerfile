FROM python:3.12-slim

LABEL org.opencontainers.image.title="Omni Slated Mem Core"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="Agent Memory Challenge Cycle 2 entry: textual track, academic division. Zero-LLM local memory pipeline."

WORKDIR /app

# Pinned, vendored wheels: the build needs no network.
COPY requirements.txt ./
COPY vendor/ ./vendor/
RUN pip install --no-cache-dir --no-index --find-links /app/vendor/wheels \
        -r requirements.txt \
    && rm -rf /app/vendor

# Application + vendored models (loaded from local disk at runtime; no downloads).
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Entrypoint: serve the platform's Add/Search contract.
# Honors $PORT when the platform injects one; defaults to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
