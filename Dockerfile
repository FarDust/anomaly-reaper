FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    UV_HTTP_TIMEOUT=120

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management (faster than pip)
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Copy source code
COPY src/ ./src/
COPY README.md ./README.md

# Create and activate virtual environment, then install dependencies
RUN uv venv /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install --system "." \
    && uv pip install --system "psycopg2-binary"

# Expose the port that the app will run on
EXPOSE 8080

# Set the entry point
ENTRYPOINT ["anomaly-reaper-api"]
CMD ["--port", "8080", "--host", "0.0.0.0"]
