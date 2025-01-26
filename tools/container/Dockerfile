# Build stage
FROM bitnami/minideb:latest AS builder

# Install system packages including Python and development tools
RUN install_packages \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    requests \
    pytest \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter \
    poetry

# Final stage
FROM bitnami/minideb:latest

# Install runtime packages only
RUN install_packages \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.* /usr/local/lib/
COPY --from=builder /usr/local/bin /usr/local/bin

# Set Python aliases
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

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