# Multi-platform digest for the official Python 3.13 slim image.
# Update the tag and digest together after reviewing upstream changes.
ARG PYTHON_BASE_IMAGE=python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285
FROM ${PYTHON_BASE_IMAGE}

ARG APP_UID=10001
ARG APP_GID=10001

# Prevent Python from writing .pyc files and ensure
# logs are immediately visible in Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HOME=/home/appuser
ENV XDG_CACHE_HOME=/home/appuser/.cache
ENV HF_HOME=/home/appuser/.cache/huggingface

RUN groupadd \
    --gid "${APP_GID}" \
    --system appuser \
    && useradd \
    --uid "${APP_UID}" \
    --gid "${APP_GID}" \
    --system \
    --create-home \
    --home-dir /home/appuser \
    --shell /usr/sbin/nologin \
    appuser

WORKDIR /app

# Install dependencies separately so Docker can cache
# this expensive layer when application code changes.
COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# Copy application source.
COPY --chown=appuser:appuser . .

RUN mkdir -p \
    /app/storage/faiss_index \
    /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser \
    /app \
    /home/appuser

# Runtime processes must not execute as root.
USER appuser:appuser

# FastAPI port.
EXPOSE 8000

# The command is overridden by Docker Compose for
# the Streamlit service.
CMD ["python", "-m", "uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
