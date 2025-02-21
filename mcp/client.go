package mcp

// import (
// 	"bytes"
// 	"context"
// 	"encoding/json"
// 	"fmt"
// 	"io"
// 	"net/http"

// 	"github.com/theapemachine/caramba/stream"
// 	"github.com/theapemachine/caramba/types"
// )

// // Client implements the Tool interface for interacting with MCP servers.
// type Client struct {
// 	conn   io.ReadWriteCloser
// 	client *http.Client
// 	config *Config
// }

// // Config holds the configuration for the MCP client.
// type Config struct {
// 	ServerURL string            `json:"server_url"`
// 	Headers   map[string]string `json:"headers"`
// }

// // NewClient creates a new MCP client with the given configuration.
// func NewClient(config *Config) *Client {
// 	return &Client{
// 		client: &http.Client{},
// 		config: config,
// 	}
// }

// // Name returns the name of the tool.
// func (c *Client) Name() string {
// 	return "mcp"
// }

// // Description returns a description of what the tool does.
// func (c *Client) Description() string {
// 	return "Interacts with MCP servers to perform various operations"
// }

// // GenerateSchema returns the JSON schema for the tool's configuration.
// func (c *Client) GenerateSchema() any {
// 	schema := `{
// 		"type": "object",
// 		"properties": {
// 			"server_url": {
// 				"type": "string",
// 				"description": "URL of the MCP server"
// 			},
// 			"headers": {
// 				"type": "object",
// 				"description": "Additional headers to include in requests",
// 				"additionalProperties": {
// 					"type": "string"
// 				}
// 			}
// 		},
// 		"required": ["server_url"]
// 	}`

// 	return schema
// }

// // Use processes the input using the tool and returns results through the accumulator.
// func (c *Client) Use(
// 	accumulator *stream.Accumulator,
// 	input map[string]any,
// 	generators ...types.Generator,
// ) *stream.Accumulator {
// 	// Convert input to JSON for sending to MCP server
// 	payload, err := json.Marshal(input)
// 	if err != nil {
// 		accumulator.Append(fmt.Sprintf("failed to marshal input: %v", err))
// 		return accumulator
// 	}

// 	// Create request to MCP server with payload
// 	req, err := http.NewRequest(http.MethodPost, c.config.ServerURL, bytes.NewBuffer(payload))
// 	if err != nil {
// 		accumulator.Append(fmt.Sprintf("failed to create request: %v", err))
// 		return accumulator
// 	}

// 	// Add headers from config
// 	for key, value := range c.config.Headers {
// 		req.Header.Add(key, value)
// 	}

// 	// Set content type header
// 	req.Header.Set("Content-Type", "application/json")

// 	// Send request
// 	resp, err := c.client.Do(req)
// 	if err != nil {
// 		accumulator.Append(fmt.Sprintf("failed to send request: %v", err))
// 		return accumulator
// 	}
// 	defer resp.Body.Close()

// 	// Read response
// 	body, err := io.ReadAll(resp.Body)
// 	if err != nil {
// 		accumulator.Append(fmt.Sprintf("failed to read response: %v", err))
// 		return accumulator
// 	}

// 	// Add response to accumulator
// 	accumulator.Append(string(body))
// 	return accumulator
// }

// // Connect establishes a connection with the MCP server.
// func (c *Client) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
// 	c.conn = conn
// 	return nil
// }
