package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stores"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

type TaskRequestError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data"`
}

// JSONRPCErrorResponse defines the standard structure for JSON-RPC errors.
type JSONRPCErrorResponse struct {
	Jsonrpc string            `json:"jsonrpc"`
	ID      string            `json:"id"` // Match the request ID if possible
	Error   *TaskRequestError `json:"error"`
}

// Manager handles task operations
type Manager struct {
	taskStore   types.Store
	llmProvider provider.ProviderType
}

// ManagerOption represents a manager configuration option
type ManagerOption func(*Manager)

// NewManager creates a new task manager with optional configuration
func NewManager(opts ...ManagerOption) *Manager {
	errnie.Trace("task.NewManager")

	manager := &Manager{}

	for _, opt := range opts {
		opt(manager)
	}

	return manager
}

// HandleTask handles incoming task requests
func (m *Manager) HandleTask(
	ctx fiber.Ctx, request *task.TaskRequest, writers io.Writer,
) error {
	switch request.Method {
	case "tasks/send":
		return m.handleTaskSend(ctx, request, writers)
	default:
		return sendJSONRPCError(
			ctx,
			request.ID,
			-32601, // Method not found
			fmt.Sprintf("method not found: %s", request.Method),
			nil,
		)
	}
}

func (manager *Manager) storeTask(t *task.Task) (err error) {
	if manager.taskStore != nil && t.ID != "" {
		session := stores.NewSession(
			stores.WithStore[*task.Task](manager.taskStore),
			stores.WithQuery[*task.Task](types.NewQuery(
				types.WithFilter("id", t.ID),
			)),
		)

		defer func() {
			if err := session.Close(); err != nil {
				errnie.New(errnie.WithError(err))
			}
		}()

		if _, err := io.Copy(session, t); err != nil {
			return err
		}
	}

	return nil
}

// handleTaskSend handles the tasks/send endpoint
func (manager *Manager) handleTaskSend(
	ctx fiber.Ctx, request *task.TaskRequest, writers io.Writer,
) error {
	errnie.Trace("task manager.handleTaskSend", "request", request)

	if err := manager.storeTask(request.Params); err != nil {
		return sendJSONRPCError(
			ctx,
			request.ID,
			-32603, // Internal error
			"Internal server error during task processing",
			err.Error(),
		)
	}

	chunks, err := manager.llmProvider.Stream(ctx, request)

	if err != nil {
		return sendJSONRPCError(
			ctx,
			request.ID,
			-32603, // Internal error
			"Internal server error during task processing",
			err.Error(),
		)
	}

	// Set up streaming response
	ctx.Set("Content-Type", "text/event-stream")
	ctx.Set("Cache-Control", "no-cache")
	ctx.Set("Connection", "keep-alive")
	ctx.Set("Transfer-Encoding", "chunked")

	go func() {
		// Check if writers is set
		if writers == nil {
			errnie.New(errnie.WithError(errors.New("writer is nil in handleTaskSend")))
			return
		}

		for chunk := range chunks {
			for _, msg := range chunk.Result.History {
				request.Params.AddMessage(msg)
			}

			if err := manager.storeTask(request.Params); err != nil {
				errnie.Warn("failed to update task in store", "error", err)
				continue
			}

			respData, err := json.Marshal(chunk)

			if err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			if _, err = fmt.Fprintf(writers, "data: %s\n\n", respData); err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			if f, ok := writers.(http.Flusher); ok {
				f.Flush()
			}
		}
	}()

	return nil
}

// sendJSONRPCError sends a JSON-RPC error response
func sendJSONRPCError(ctx fiber.Ctx, id string, code int, message string, data any) error {
	httpStatus := http.StatusInternalServerError
	switch code {
	case -32700, -32600, -32601, -32602: // Parse error, Invalid request, Method not found, Invalid params
		httpStatus = http.StatusBadRequest
	case -32001: // Task not found
		httpStatus = http.StatusNotFound
	}

	errResponse := JSONRPCErrorResponse{
		Jsonrpc: "2.0",
		ID:      id,
		Error: &TaskRequestError{
			Code:    code,
			Message: message,
			Data:    fmt.Sprintf("%v", data),
		},
	}

	errnie.Warn("Sending JSON-RPC Error", "code", code, "message", message, "data", data, "httpStatus", httpStatus)

	return ctx.Status(httpStatus).JSON(errResponse)
}

// WithTaskStore sets the task store
func WithTaskStore(store types.Store) ManagerOption {
	return func(m *Manager) {
		m.taskStore = store
	}
}

// WithLLMProvider sets the LLM provider
func WithLLMProvider(provider provider.ProviderType) ManagerOption {
	return func(m *Manager) {
		m.llmProvider = provider
	}
}
