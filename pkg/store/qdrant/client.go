package qdrant

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"

	qc "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/config"
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
ConfigFromEnv loads store.qdrant.* from config.yml (see pkg/config).

When a URL uses port 6333 (the common REST port in examples), the gRPC port defaults to 6334 unless
overridden by grpc_port / port. Scheme https sets UseTLS to true.
*/
func ConfigFromEnv() Config {
	return configFromApp(config.NewQdrantStoreConfig())
}

func configFromApp(appConfig config.QdrantStoreConfig) Config {
	rawURL := strings.TrimSpace(appConfig.URL)

	if rawURL == "" {
		rawURL = strings.TrimSpace(appConfig.BaseURL)
	}

	port := appConfig.GRPCPort

	if port == 0 {
		port = appConfig.Port
	}

	host, port, useTLS := mergeURLOverrides(
		rawURL,
		strings.TrimSpace(appConfig.Host),
		port,
		appConfig.UseTLS,
	)

	return Config{
		Host:     host,
		Port:     port,
		APIKey:   strings.TrimSpace(appConfig.APIKey),
		UseTLS:   useTLS,
		PoolSize: uint(appConfig.PoolSize),
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
