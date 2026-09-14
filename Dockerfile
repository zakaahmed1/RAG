FROM python:3.13-slim

# Prevent Python from writing .pyc files and ensure
# logs are immediately visible in Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies separately so Docker can cache
# this expensive layer when application code changes.
COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip \
    && python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# Copy application source.
COPY . .

# FastAPI port.
EXPOSE 8000

# The command is overridden by Docker Compose for
# the Streamlit service.
CMD ["python", "-m", "uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]