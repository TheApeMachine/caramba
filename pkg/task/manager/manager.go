package manager

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gofiber/fiber/v3"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/task"
)

// JSONRPCErrorResponse defines the standard structure for JSON-RPC errors.
type JSONRPCErrorResponse struct {
	Jsonrpc string                 `json:"jsonrpc"`
	ID      string                 `json:"id"` // Match the request ID if possible
	Error   *task.TaskRequestError `json:"error"`
}

type Manager struct {
	taskStore   task.TaskStore
	llmProvider provider.ProviderType
}

type ManagerOption func(*Manager)

func NewManager(opts ...ManagerOption) *Manager {
	errnie.Trace("task.NewManager")

	manager := &Manager{}

	for _, opt := range opts {
		opt(manager)
	}

	return manager
}

func (manager *Manager) HandleTask(ctx fiber.Ctx, request *task.TaskRequest) error {
	switch request.Method {
	case "tasks/send":
		return manager.handleTaskSend(ctx, request)
	default:
		return sendJSONRPCError(
			ctx,
			request.ID,
			task.ErrorMethodNotFound,
			fmt.Sprintf("method not found: %s", request.Method),
			nil,
		)
	}
}

func (manager *Manager) handleTaskSend(ctx fiber.Ctx, request *task.TaskRequest) error {
	errnie.Trace("task manager.handleTaskSend", "request", request)

	chunks, err := manager.llmProvider.Generate(ctx, request)

	if err != nil {
		internalErr := errnie.New(errnie.WithError(err))
		errnie.Warn("LLM generation error", "error", internalErr)

		return sendJSONRPCError(
			ctx,
			request.ID,
			task.ErrorInternalError,
			"Internal server error during task processing",
			internalErr.Error(),
		)
	}

	return ctx.SendStreamWriter(func(w *bufio.Writer) {
		for chunk := range chunks {
			buf, err := json.Marshal(chunk)
			if err != nil {
				errnie.Error("Error marshalling stream chunk", "error", err)
				return // Stop sending on marshalling error
			}

			if _, err := w.Write(buf); err != nil {
				errnie.Error("Error writing to stream", "error", err)
				return // Stop sending on write error
			}

			if err := w.Flush(); err != nil {
				errnie.Error("Error flushing stream writer", "error", err)
				return // Stop sending on flush error
			}
		}
	})
}

// sendJSONRPCError is a helper to format and send JSON-RPC errors via Fiber.
func sendJSONRPCError(ctx fiber.Ctx, requestID string, code int, message string, data any) error {
	httpStatus := http.StatusInternalServerError // Default to 500
	switch code {
	case task.ErrorParseError, task.ErrorInvalidRequest, task.ErrorMethodNotFound, task.ErrorInvalidParams:
		httpStatus = http.StatusBadRequest // 400
	case task.ErrorTaskNotFound:
		httpStatus = http.StatusNotFound // 404
		// Add other mappings as needed
	}

	errResponse := JSONRPCErrorResponse{
		Jsonrpc: "2.0",
		ID:      requestID, // Echo the request ID
		Error: &task.TaskRequestError{
			Code:    code,
			Message: message,
			Data:    fmt.Sprintf("%v", data), // Convert data to string
		},
	}

	errnie.Warn("Sending JSON-RPC Error", "code", code, "message", message, "data", data, "httpStatus", httpStatus)

	return ctx.Status(httpStatus).JSON(errResponse)
}

func WithTaskStore(taskStore task.TaskStore) ManagerOption {
	return func(manager *Manager) {
		manager.taskStore = taskStore
	}
}

func WithLLMProvider(llmProvider provider.ProviderType) ManagerOption {
	return func(manager *Manager) {
		manager.llmProvider = llmProvider
	}
}
