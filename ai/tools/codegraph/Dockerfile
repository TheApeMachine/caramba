FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY ai/tools/codegraph/requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY ai/tools/codegraph/ ./ai/tools/codegraph/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser
RUN chown -R appuser:appuser /app

EXPOSE 8001

# Healthcheck to probe service readiness
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

USER appuser

CMD ["python", "-m", "ai.tools.codegraph.tool"]
