package client

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/sourcegraph/jsonrpc2"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/jsonrpc"
	"github.com/theapemachine/caramba/pkg/task"
)

// RPCClient handles JSON-RPC 2.0 communication with the server
type RPCClient struct {
	conn    *jsonrpc2.Conn
	baseURL string
}

// RPCClientOption defines functional options for configuring the RPCClient
type RPCClientOption func(*RPCClient)

// NewRPCClient creates a new JSON-RPC 2.0 client instance with retry logic
func NewRPCClient(opts ...RPCClientOption) (*RPCClient, error) {
	client := &RPCClient{}

	for _, opt := range opts {
		opt(client)
	}

	// Validate required configuration
	if client.baseURL == "" {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("baseURL is required")))
	}

	conn, err := net.DialTimeout("tcp", client.baseURL, 10*time.Second)

	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	stream := jsonrpc2.NewBufferedStream(conn, jsonrpc2.VSCodeObjectCodec{})
	client.conn = jsonrpc2.NewConn(
		context.Background(),
		stream,
		jsonrpc2.HandlerWithError(client.handle),
	)

	return client, nil
}

// WithBaseURL sets the base URL for the RPC client
func WithBaseURL(url string) RPCClientOption {
	return func(c *RPCClient) {
		c.baseURL = url
	}
}

// SendTask sends a task to the RPC server
func (c *RPCClient) SendTask(req *task.TaskRequest, writer any) (*task.TaskResponse, error) {
	errnie.Trace("client.SendTask")

	if c == nil || c.conn == nil {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("client not properly initialized")))
	}

	if req == nil {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("invalid task request: request is nil")))
	}

	const method = "tasks/send"

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var response jsonrpc.Response

	if err := c.conn.Call(ctx, method, req.Params, &response); err != nil {
		if rpcErr, ok := err.(*jsonrpc2.Error); ok {
			return nil, errnie.New(errnie.WithError(fmt.Errorf("RPC error: code %d: %s", rpcErr.Code, rpcErr.Message)))
		}
		return nil, errnie.New(errnie.WithError(fmt.Errorf("RPC call failed: %w", err)))
	}

	if out, ok := response.Result.(task.TaskResponse); ok {
		return &out, nil
	}

	return nil, errnie.New(errnie.WithError(fmt.Errorf("invalid response type: expected TaskResponse, got %T", response.Result)))
}

// SendTaskStream sends a task to the RPC server using the A2A protocol's streaming approach via SSE
// It returns a channel that delivers incremental updates as they arrive via Server-Sent Events
func (c *RPCClient) SendTaskStream(req *task.TaskRequest) (<-chan *task.TaskResponse, error) {
	if c == nil || c.conn == nil {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("client not properly initialized")))
	}

	// Validate task request pointer
	if req == nil {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("invalid task request: request is nil")))
	}

	// Create output channel for streaming responses
	responseChan := make(chan *task.TaskResponse, 10) // Buffer of 10 to prevent blocking

	// Extract the base address from the connection for making HTTP requests
	// This assumes the baseURL is in the format "host:port" and we need to convert to "http://host:port"
	baseURL := "http://" + c.baseURL

	// Construct the SSE endpoint URL
	sseEndpoint := fmt.Sprintf("%s/api/tasks/stream", baseURL)

	// Context for the HTTP request with a timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)

	// Initiate the subscription using Notify (doesn't wait for response)
	go func() {
		// Use Notify to initiate the subscription without waiting for a response
		if err := c.conn.Notify(ctx, "tasks/sendSubscribe", req.Params); err != nil {
			errnie.New(errnie.WithError(err))
			errorResponse := task.NewTaskResponse(task.WithResponseError(err))
			responseChan <- errorResponse
			close(responseChan)
			cancel()
			return
		}

		// Create the HTTP request
		httpReq, err := http.NewRequestWithContext(ctx, "GET", sseEndpoint, nil)
		if err != nil {
			errorResponse := task.NewTaskResponse(task.WithResponseError(err))
			responseChan <- errorResponse
			close(responseChan)
			cancel()
			return
		}

		// Set appropriate headers
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("Cache-Control", "no-cache")
		httpReq.Header.Set("Connection", "keep-alive")
		// Add task ID as a query parameter for correlation
		q := httpReq.URL.Query()
		q.Add("taskId", req.Params.ID)
		httpReq.URL.RawQuery = q.Encode()

		// Make the HTTP request
		httpClient := &http.Client{}
		resp, err := httpClient.Do(httpReq)
		if err != nil {
			errnie.New(errnie.WithError(err))
			errorResponse := task.NewTaskResponse(task.WithResponseError(err))
			responseChan <- errorResponse
			close(responseChan)
			cancel()
			return
		}
		defer resp.Body.Close()

		// Check if the response is successful
		if resp.StatusCode != http.StatusOK {
			errnie.New(errnie.WithError(fmt.Errorf("unexpected HTTP status: %s", resp.Status)))
			errorResponse := task.NewTaskResponse(task.WithResponseError(
				errnie.New(errnie.WithError(fmt.Errorf("unexpected HTTP status: %s", resp.Status))),
			))
			responseChan <- errorResponse
			close(responseChan)
			cancel()
			return
		}

		errnie.Info("Started SSE stream connection")

		// Create a Scanner to read the SSE stream line by line
		scanner := bufio.NewScanner(resp.Body)
		var messageBuffer strings.Builder

		// Process the SSE stream
		for scanner.Scan() {
			line := scanner.Text()

			// SSE messages start with "data: " and end with an empty line
			if strings.HasPrefix(line, "data: ") {
				// Extract the JSON payload
				jsonData := line[6:] // Skip "data: " prefix
				messageBuffer.WriteString(jsonData)
			} else if line == "" && messageBuffer.Len() > 0 {
				// Empty line indicates the end of a message
				jsonData := messageBuffer.String()
				messageBuffer.Reset()

				// Parse the JSON payload into a TaskResponse
				var response task.TaskResponse
				if err := json.Unmarshal([]byte(jsonData), &response); err != nil {
					errnie.New(errnie.WithError(err))
					continue
				}

				// Send the response through the channel
				responseChan <- &response

				// Check if this is the final message
				if response.Result != nil &&
					response.Result.Status.State == task.TaskStateCompleted {
					errnie.Debug("Received final message in stream")
					// Final message received, we can exit
					cancel()
					return
				}
			}

			// Check if the context has been canceled or timed out
			select {
			case <-ctx.Done():
				errnie.Debug("Context done, stopping SSE processing", "error", ctx.Err())
				if ctx.Err() == context.DeadlineExceeded {
					errorResponse := task.NewTaskResponse(task.WithResponseError(
						errnie.New(errnie.WithError(fmt.Errorf("streaming timeout exceeded"))),
					))
					responseChan <- errorResponse
				}
				return
			default:
				// Continue processing
			}
		}

		// Check for scanner errors
		if err := scanner.Err(); err != nil {
			errnie.New(errnie.WithError(err))
			errorResponse := task.NewTaskResponse(task.WithResponseError(err))
			responseChan <- errorResponse
		}
	}()

	return responseChan, nil
}

// handle implements the client-side message handler
func (c *RPCClient) handle(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) (any, error) {
	// Handle any server-initiated requests or notifications here
	return nil, nil
}
