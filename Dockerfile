FROM golang:1.26.3-trixie AS builder

WORKDIR /app/caramba

COPY go.mod go.sum ./

COPY --from=hf go.mod go.sum /app/hf/
COPY --from=manifesto go.mod go.sum /app/manifesto/
COPY --from=puter go.mod go.sum /app/puter/

RUN go mod download && go mod verify

COPY . .

COPY --from=hf / /app/hf/
COPY --from=manifesto / /app/manifesto/
COPY --from=puter / /app/puter/

RUN go build -ldflags='-checklinkname=0' -o main main.go

EXPOSE 8118

CMD ["./main", "serve"]
