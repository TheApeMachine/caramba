FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
# Expected SHA256 checksum for uv installer (update this when installer changes)
# To get current checksum: curl -LsSf https://astral.sh/uv/install.sh | sha256sum
ARG UV_INSTALLER_SHA256=""
RUN set -eux; \
    INSTALLER="/tmp/uv-installer.sh"; \
    curl -LsSf -o "$INSTALLER" https://astral.sh/uv/install.sh; \
    if [ -n "$UV_INSTALLER_SHA256" ]; then \
        echo "$UV_INSTALLER_SHA256  $INSTALLER" | sha256sum -c - || { echo "Installer checksum mismatch!" >&2; exit 1; }; \
    fi; \
    sh "$INSTALLER"; \
    rm "$INSTALLER"

# Make uv available for subsequent build steps.
RUN mv /root/.local/bin/uv /usr/local/bin/uv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser -s /bin/bash appuser

# Add uv to PATH (for appuser)
ENV PATH="/usr/local/bin:${PATH}"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Set up working directory
WORKDIR /app

# Copy tool requirements (no uv.lock needed)
COPY ai/tools/webcrawl/requirements.txt ./requirements.txt

# Install Python dependencies (cache in user's home)
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy application code
COPY ai/tools/webcrawl/ ./ai/tools/webcrawl/

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose MCP server port
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8001/health || exit 1

# Run the MCP server
CMD ["python", "-m", "ai.tools.webcrawl.tool", "--host", "0.0.0.0", "--port", "8001"]