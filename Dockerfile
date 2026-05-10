FROM golang:1.26.3-trixie AS builder

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod download && go mod verify

COPY . .

RUN go build -o main main.go

EXPOSE 8118

CMD ["./main"]