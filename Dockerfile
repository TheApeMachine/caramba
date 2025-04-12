FROM bitnami/minideb:latest AS builder

RUN install_packages \
    ca-certificates \
    curl

ENV GOVERSION=1.24.2
ARG TARGETARCH
RUN curl -LO https://golang.org/dl/go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    tar -C /usr/local -xzf go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    rm go${GOVERSION}.linux-${TARGETARCH}.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOFLAGS=-buildvcs=false

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# ENV GOCACHE=/root/.cache/go-build
# RUN --mount=type=cache,target="/root/.cache/go-build" go build -o main
RUN go build -o main

FROM bitnami/minideb:latest

WORKDIR /etc/ssl/certs

COPY --from=builder /etc/ssl/certs/ca-certificates.crt .

WORKDIR /app

COPY --from=builder /app/main .

CMD ["./main"]
