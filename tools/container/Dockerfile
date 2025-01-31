# Build stage
FROM bitnami/minideb:latest

# Install system packages including Python and development tools
RUN install_packages \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages in virtual environment
RUN pip3 install --no-cache-dir \
    requests \
    pytest \
    black \
    flake8 \
    mypy \
    poetry

# Create non-root user
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user/.ssh && \
    chown -R user:user /home/user

# Set up SSH for the new user
WORKDIR /home/user/.ssh
COPY --chown=user:user id_marvin id_marvin.pub ./
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /home/user/.ssh/config && \
    chmod 600 id_marvin && \
    chmod 644 id_marvin.pub && \
    chmod 644 config

WORKDIR /home/user
USER user
