package qdrant

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"

	qc "github.com/qdrant/go-client/qdrant"
)

const (
	defaultGRPCHost = "localhost"
	defaultGRPCPort = 6334
)

/*
Config holds gRPC connection settings for Qdrant via the official go-client.

Host defaults to localhost; Port defaults to 6334 (gRPC). Docker Compose typically maps REST on 6333
and gRPC on 6334 — set Port accordingly. UseTLS enables TLS (for example Qdrant Cloud); APIKey is
sent on the gRPC connection per client configuration.

PoolSize is passed through to the underlying client; 0 lets the library default apply (see
qdrant.NewClient).
*/
type Config struct {
	Host     string
	Port     int
	APIKey   string
	UseTLS   bool
	PoolSize uint
}

/*
Client wraps github.com/qdrant/go-client/qdrant for mosaic. Close the client when shutting down.
*/
type Client struct {
	inner *qc.Client
}

/*
NewClient connects to Qdrant using the official gRPC client.
*/
func NewClient(cfg Config) (*Client, error) {
	host := strings.TrimSpace(cfg.Host)

	if host == "" {
		host = defaultGRPCHost
	}

	port := cfg.Port

	if port == 0 {
		port = defaultGRPCPort
	}

	inner, err := qc.NewClient(&qc.Config{
		Host:     host,
		Port:     port,
		APIKey:   strings.TrimSpace(cfg.APIKey),
		UseTLS:   cfg.UseTLS,
		PoolSize: cfg.PoolSize,
	})

	if err != nil {
		return nil, fmt.Errorf("qdrant: new client: %w", err)
	}

	return &Client{inner: inner}, nil
}

/*
ConfigFromEnv loads QDRANT_HOST, QDRANT_GRPC_PORT or QDRANT_PORT, QDRANT_API_KEY, QDRANT_USE_TLS,
QDRANT_POOL_SIZE, and optionally QDRANT_URL or QDRANT_BASE_URL.

When a URL uses port 6333 (the common REST port in examples), the gRPC port defaults to 6334 unless
overridden by QDRANT_GRPC_PORT / QDRANT_PORT. Scheme https sets UseTLS to true.
*/
func ConfigFromEnv() Config {
	rawURL := strings.TrimSpace(os.Getenv("QDRANT_URL"))

	if rawURL == "" {
		rawURL = strings.TrimSpace(os.Getenv("QDRANT_BASE_URL"))
	}

	host := strings.TrimSpace(os.Getenv("QDRANT_HOST"))
	port := envIntDefault0("QDRANT_GRPC_PORT")

	if port == 0 {
		port = envIntDefault0("QDRANT_PORT")
	}

	useTLS := strings.EqualFold(strings.TrimSpace(os.Getenv("QDRANT_USE_TLS")), "true")
	poolSize := uint(envIntDefault0("QDRANT_POOL_SIZE"))

	host, port, useTLS = mergeURLOverrides(rawURL, host, port, useTLS)

	return Config{
		Host:     host,
		Port:     port,
		APIKey:   os.Getenv("QDRANT_API_KEY"),
		UseTLS:   useTLS,
		PoolSize: poolSize,
	}
}

/*
NewClientFromEnv is equivalent to NewClient(ConfigFromEnv()).
*/
func NewClientFromEnv() (*Client, error) {
	return NewClient(ConfigFromEnv())
}

/*
Native returns the underlying go-client for advanced APIs (aliases, snapshots, raw gRPC).
*/
func (client *Client) Native() *qc.Client {
	return client.inner
}

/*
Close releases all pooled gRPC connections.
*/
func (client *Client) Close() error {
	if err := client.inner.Close(); err != nil {
		return fmt.Errorf("qdrant: close: %w", err)
	}

	return nil
}

func mergeURLOverrides(rawURL, host string, port int, useTLS bool) (string, int, bool) {
	if strings.TrimSpace(rawURL) == "" {
		if host == "" {
			host = defaultGRPCHost
		}

		if port == 0 {
			port = defaultGRPCPort
		}

		return host, port, useTLS
	}

	parsed, err := url.Parse(rawURL)

	if err != nil || parsed.Hostname() == "" {
		if host == "" {
			host = defaultGRPCHost
		}

		if port == 0 {
			port = defaultGRPCPort
		}

		return host, port, useTLS
	}

	if host == "" {
		host = parsed.Hostname()
	}

	if parsed.Scheme == "https" {
		useTLS = true
	}

	if port == 0 && parsed.Port() != "" {
		if wirePort, err := strconv.Atoi(parsed.Port()); err == nil {
			if wirePort == 6333 {
				port = defaultGRPCPort
			} else {
				port = wirePort
			}
		}
	}

	if port == 0 {
		port = defaultGRPCPort
	}

	return host, port, useTLS
}

func envIntDefault0(key string) int {
	raw := strings.TrimSpace(os.Getenv(key))

	if raw == "" {
		return 0
	}

	v, err := strconv.Atoi(raw)

	if err != nil {
		return 0
	}

	return v
}
