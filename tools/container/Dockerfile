# Base stage
FROM bitnami/minideb:latest AS base

# Install a comprehensive set of development tools and languages.
# This allows for a versatile development environment capable of
# handling a wide range of programming tasks.
RUN install_packages \
    ca-certificates \
    wget \
    curl \
    git \
    jq \
    unzip \
    zip


WORKDIR /root/.ssh
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

WORKDIR /root