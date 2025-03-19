# Base stage
FROM bitnami/minideb:latest

ENV USER=developer

# Install a comprehensive set of development tools and languages.
# This allows for a versatile development environment capable of
# handling a wide range of programming tasks.
RUN install_packages \
    sudo \
    ca-certificates \
    wget \
    curl \
    git \
    jq \
    unzip \
    zip \
    nmap \
    net-tools \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    python3-distutils \
    && useradd -m ${USER} -s /bin/bash && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER}

WORKDIR /root/.ssh
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config
WORKDIR /home/${USER}
RUN chown -R ${USER}:${USER} /home/${USER}
USER ${USER}
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"