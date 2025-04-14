FROM bitnami/minideb:latest

RUN install_packages \
    ca-certificates \
    curl

ENV GOVERSION=1.24.2
ARG TARGETARCH
RUN curl -LO https://golang.org/dl/go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    tar -C /usr/local -xzf go${GOVERSION}.linux-${TARGETARCH}.tar.gz && \
    rm go${GOVERSION}.linux-${TARGETARCH}.tar.gz

ENV PATH="/usr/local/go/bin:/usr/bin:/usr/local/go/bin:/root/go/bin:${PATH}"
ENV GOFLAGS=-buildvcs=false

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go install github.com/air-verse/air@latest

COPY . .

# ENV GOCACHE=/root/.cache/go-build
# RUN --mount=type=cache,target="/root/.cache/go-build" go build -o main
RUN go build -o main

CMD ["air"]
