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
		return nil, errnie.New(
			errnie.WithType(errnie.ValidationError),
			errnie.WithError(fmt.Errorf("invalid task request: request is nil")),
		)
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
	ch := make(chan *task.TaskResponse)

	// Compose the SSE endpoint URL (assume /task/:id/stream)
	if req == nil || req.Params == nil || req.Params.ID == "" {
		close(ch)
		return ch, fmt.Errorf("invalid task request: missing task ID")
	}

	// Derive base HTTP URL from baseURL (strip port if needed)
	base := c.baseURL
	if !strings.HasPrefix(base, "http") {
		base = "http://" + base
	}
	url := fmt.Sprintf("%s/task/%s/stream", base, req.Params.ID)

	// Marshal the task request in case the SSE endpoint expects POST body (optional)
	go func() {
		defer close(ch)
		resp, err := http.Get(url)
		if err != nil {
			errnie.New(errnie.WithError(fmt.Errorf("failed to connect to SSE endpoint: %w", err)))
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data:") {
				payload := strings.TrimSpace(line[5:])
				if payload == "" {
					continue
				}
				var respObj task.TaskResponse
				if err := json.Unmarshal([]byte(payload), &respObj); err == nil {
					ch <- &respObj
					if respObj.Result != nil && respObj.Result.Status.State == task.TaskStateCompleted {
						break
					}
				}
			}
		}
	}()

	return ch, nil
}

// handle implements the client-side message handler
func (c *RPCClient) handle(ctx context.Context, conn *jsonrpc2.Conn, req *jsonrpc2.Request) (any, error) {
	// Handle any server-initiated requests or notifications here
	return nil, nil
}
