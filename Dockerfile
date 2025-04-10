FROM bitnami/minideb:latest AS base

RUN install_packages \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libsodium-dev \
    libczmq-dev \
    libczmq4

ENV GOVERSION=1.24.2
ARG TARGETARCH
RUN curl -LO https://golang.org/dl/go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    tar -C /usr/local -xzf go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    rm go${GOVERSION}.linux-${TARGETARCH}.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOFLAGS=-buildvcs=false
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/include"
ENV CGO_LDFLAGS="-L/usr/lib"

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod tidy && go mod download

COPY . .
RUN go build -tags cgo -o main

CMD ["/app/main"]
