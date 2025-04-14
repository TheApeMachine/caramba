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

	// Retry connection with exponential backoff
	maxRetries := 5
	var lastErr error
	for i := range maxRetries {
		if i > 0 {
			time.Sleep(time.Duration(i*i) * time.Second)
		}

		// Establish connection
		conn, err := net.DialTimeout("tcp", client.baseURL, 10*time.Second)
		if err != nil {
			lastErr = err
			errnie.Warn(
				"connection attempt failed",
				"attempt", i+1,
				"error", err,
			)
			continue
		}

		// Create JSON-RPC 2.0 connection
		stream := jsonrpc2.NewBufferedStream(conn, jsonrpc2.VSCodeObjectCodec{})
		client.conn = jsonrpc2.NewConn(
			context.Background(),
			stream,
			jsonrpc2.HandlerWithError(client.handle),
		)

		// Connection successful
		errnie.Info(
			"Successfully connected to RPC server",
			"baseURL", client.baseURL,
		)
		return client, nil
	}

	return nil, errnie.New(errnie.WithError(fmt.Errorf("failed to connect to RPC server after %d attempts: %w", maxRetries, lastErr)))
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

	// Validate task request pointer
	if req == nil {
		return nil, errnie.New(errnie.WithError(fmt.Errorf("invalid task request: request is nil")))
	}

	// The method name for the RPC call
	const method = "tasks/send"

	// Context for the RPC call
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Prepare the variable to hold the result (the Task struct)
	var resultTask task.Task

	// Make the RPC call
	// Pass the task parameters (req.Params) directly.
	// Pass the address of resultTask to unmarshal the JSON-RPC result into it.
	if err := c.conn.Call(ctx, method, req.Params, &resultTask); err != nil {
		// Handle potential jsonrpc2.Error for structured errors
		if rpcErr, ok := err.(*jsonrpc2.Error); ok {
			return nil, errnie.New(errnie.WithError(fmt.Errorf("RPC error: code %d: %s", rpcErr.Code, rpcErr.Message)))
		}
		// Handle other potential errors (e.g., connection issues)
		return nil, errnie.New(errnie.WithError(fmt.Errorf("RPC call failed: %w", err)))
	}

	// If the call was successful, the resultTask should be populated.
	// Create the TaskResponse.
	response := task.NewTaskResponse()
	response.Result = &resultTask

	return response, nil
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
func (c *RPCClient) handle(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) (interface{}, error) {
	// Handle any server-initiated requests or notifications here
	return nil, nil
}
