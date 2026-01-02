FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
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

# Set up working directory
WORKDIR /app

# Copy tool requirements (no uv.lock needed)
COPY ai/tools/deeplake/requirements.txt ./requirements.txt

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Ensure git is available for GitPython/colbert.
RUN git --version

# Fail the image build early if key deps didn't install correctly.
RUN python -c "import deeplake, torch; print('deeplake', deeplake.__version__, 'torch', torch.__version__)"

# Copy application code
COPY ai/tools/deeplake/ ./ai/tools/deeplake/

# Expose MCP server port
EXPOSE 8001

# Run the MCP server
CMD ["python", "-m", "ai.tools.deeplake.tool", "--host", "0.0.0.0", "--port", "8001"]