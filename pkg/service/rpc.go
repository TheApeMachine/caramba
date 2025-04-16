package service

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/sourcegraph/jsonrpc2"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/errnie"
	pkgerr "github.com/theapemachine/caramba/pkg/errors"
	"github.com/theapemachine/caramba/pkg/jsonrpc"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/valyala/fasthttp"
)

// Handler implements the JSON-RPC 2.0 message handler
type Handler struct {
	agent agent.Agent
	mu    sync.RWMutex
}

// NewHandler creates a new RPC handler with the given A2A service
func NewHandler(agent agent.Agent) *Handler {
	log.Printf("Creating new RPC handler")
	return &Handler{
		agent: agent,
	}
}

// Handle processes incoming JSON-RPC 2.0 requests and notifications
func (h *Handler) Handle(
	ctx context.Context,
	conn *jsonrpc2.Conn,
	req *jsonrpc2.Request,
) (any, error) {
	errnie.Trace("service.Handle")

	h.mu.Lock()
	defer h.mu.Unlock()

	if req == nil {
		errnie.New(errnie.WithError(errors.New("request is nil")))

		return nil, &jsonrpc2.Error{
			Code:    jsonrpc2.CodeInvalidRequest,
			Message: "request is nil",
		}
	}

	if req.Params == nil {
		return nil, &jsonrpc2.Error{
			Code:    jsonrpc2.CodeInvalidParams,
			Message: "params is required",
		}
	}

	switch req.Method {
	case "tasks/send":
		app := fiber.New()
		c := app.AcquireCtx(&fasthttp.RequestCtx{})
		defer app.ReleaseCtx(c)

		taskRequest := task.NewTaskRequest(nil)

		if err := json.Unmarshal(*req.Params, taskRequest); err != nil {
			return nil, errnie.New(errnie.WithError(fmt.Errorf("failed to parse task params: %w", err)))
		}

		if err := h.agent.HandleTask(c, taskRequest); err != nil {
			return nil, &jsonrpc2.Error{
				Code:    jsonrpc2.CodeInternalError,
				Message: fmt.Sprintf("failed to handle task: %v", err),
			}
		}

		return taskRequest, nil
	case "tasks/sendSubscribe":

	default:
		// Return a jsonrpc2 error for unknown methods
		return nil, &jsonrpc2.Error{
			Code:    jsonrpc2.CodeMethodNotFound,
			Message: fmt.Sprintf("method not found: %s", req.Method),
		}
	}

	return nil, nil
}

// validateTaskRequest validates the task request
func (h *Handler) validateTaskRequest(req *task.TaskRequest) error {
	errnie.Trace("service.validateTaskRequest")

	if req == nil {
		return errnie.New(errnie.WithError(errors.New("task request is nil")))
	}
	// Add more specific validation for req.Params (the Task) if needed
	if req.Params.ID == "" {
		return errors.New("task ID is required")
	}
	return nil
}

// HandleSSEStream handles streaming responses via Server-Sent Events
// This is required by the A2A protocol for proper streaming support
func (h *Handler) HandleSSEStream(ctx fiber.Ctx) error {
	// Extract task ID from query params
	taskID := ctx.Query("taskId")
	if taskID == "" {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "missing taskId query parameter",
		})
	}

	// Create a task with the specified ID
	taskParams := &task.Task{ID: taskID}
	taskReq := task.NewTaskRequest(taskParams)

	if err := h.validateTaskRequest(taskReq); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": fmt.Sprintf("invalid task request: %v", err),
		})
	}

	// Set headers for SSE
	ctx.Set("Content-Type", "text/event-stream")
	ctx.Set("Cache-Control", "no-cache")
	ctx.Set("Connection", "keep-alive")
	ctx.Set("Transfer-Encoding", "chunked")

	// Set up a custom Fiber middleware to capture the response
	return ctx.SendStreamWriter(func(w *bufio.Writer) {
		// Add writer to the context
		h.agent.AddWriter(w)

		// Handle the task - this will send SSE data through the context
		if err := h.agent.HandleTask(ctx, taskReq); err != nil {
			// If there's an error, write an error message to the stream
			errorResp := task.TaskResponse{
				Response: jsonrpc.Response{
					Error: &pkgerr.JSONRPCError{
						Code:    -32000,
						Message: fmt.Sprintf("failed to handle task: %v", err),
					},
				},
			}

			data, _ := json.Marshal(errorResp)
			fmt.Fprintf(w, "data: %s\n\n", data)
			w.Flush()
			return
		}

		// The task handler will write to the response context directly
		// We just need to ensure proper flushing
		w.Flush()
	})
}

// StartRPCServer starts a JSON-RPC 2.0 server on the specified address
// and listens for context cancellation to shut down gracefully.
func (a2a *A2A) StartRPCServer(ctx context.Context, addr string) error {
	errnie.Trace("service.StartRPCServer")

	listener, err := net.Listen("tcp", addr)

	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	// Defer closing the listener.
	defer listener.Close()

	// Create a goroutine to handle context cancellation and close the listener.
	go func() {
		<-ctx.Done()
		listener.Close() // Close the listener to stop the Accept loop
	}()

	handler := NewHandler(a2a.agent)

	conns := make(chan *jsonrpc2.Conn)
	defer close(conns)

	// Connection monitor goroutine
	go func() {
		activeConns := make(map[*jsonrpc2.Conn]struct{})
		for conn := range conns {
			activeConns[conn] = struct{}{}
			go func(c *jsonrpc2.Conn) {
				<-c.DisconnectNotify()
				delete(activeConns, c)
			}(conn)
		}
	}()

	for {
		conn, err := listener.Accept()

		if err != nil {
			// Check if the error is due to the listener being closed.
			if errors.Is(err, net.ErrClosed) {
				return nil // Expected error on shutdown
			}
			if ne, ok := err.(net.Error); ok && ne.Temporary() {
				time.Sleep(100 * time.Millisecond)
				continue
			}

			return errnie.New(errnie.WithError(err))
		}

		// Create JSON-RPC stream and connection
		// Use the background context for individual connections for now.
		// If connection-specific cancellation is needed, this could be adapted.
		stream := jsonrpc2.NewBufferedStream(conn, jsonrpc2.VSCodeObjectCodec{})
		jsonConn := jsonrpc2.NewConn(context.Background(), stream, jsonrpc2.HandlerWithError(handler.Handle))

		// Track the connection
		conns <- jsonConn
	}
}
