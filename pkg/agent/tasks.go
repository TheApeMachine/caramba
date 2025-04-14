package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
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
	taskStore   task.TaskStore
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

// handleTaskSend handles the tasks/send endpoint
func (m *Manager) handleTaskSend(
	ctx fiber.Ctx, request *task.TaskRequest, writers io.Writer,
) error {
	errnie.Trace("task manager.handleTaskSend", "request", request)

	// Start streaming response from LLM provider
	chunks, err := m.llmProvider.Stream(ctx, request)
	if err != nil {
		internalErr := errnie.New(errnie.WithError(err))
		errnie.Warn("LLM generation error", "error", internalErr)

		return sendJSONRPCError(
			ctx,
			request.ID,
			-32603, // Internal error
			"Internal server error during task processing",
			internalErr.Error(),
		)
	}

	// Create initial response
	sessionID := uuid.New().String()
	response := task.NewTaskResponse(
		task.WithResponseID(request.ID),
		task.WithResponseTask(task.Task{
			ID:        request.Params.ID,
			SessionID: &sessionID,
			Status: task.TaskStatus{
				State:     task.TaskStateWorking,
				Timestamp: time.Now(),
			},
			History:   request.Params.History,
			Artifacts: make([]task.Artifact, 0),
			Metadata:  request.Params.Metadata,
		}),
	)

	// Set up streaming response
	ctx.Set("Content-Type", "text/event-stream")
	ctx.Set("Cache-Control", "no-cache")
	ctx.Set("Connection", "keep-alive")
	ctx.Set("Transfer-Encoding", "chunked")

	go func() {
		// Check if writers is set
		if writers == nil {
			errnie.New(errnie.WithError(errors.New("Writer is nil in handleTaskSend")))
			return
		}

		name := "response"

		for chunk := range chunks {
			// Debug log the chunk reception
			// We use a custom string to avoid calling methods that might not exist
			errnie.Debug("Received content chunk")

			// Add message to result history
			if len(chunk.Result.History) > 0 {
				lastMsg := chunk.Result.History[len(chunk.Result.History)-1]
				response.Result.Artifacts = append(response.Result.Artifacts, task.Artifact{
					Name: &name,
					Parts: []task.Part{
						&task.TextPart{
							Type: "text",
							Text: lastMsg.String(),
						},
					},
				})
			}

			if chunk.Result.Status.State == task.TaskStateCompleted {
				response.Result.Status.State = task.TaskStateCompleted
			}

			// Marshal the response to JSON
			respData, err := json.Marshal(response)
			if err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			// Write in SSE format
			_, err = fmt.Fprintf(writers, "data: %s\n\n", respData)
			if err != nil {
				errnie.New(errnie.WithError(err))
				return
			}

			// Try to flush if the writer supports it
			if f, ok := writers.(http.Flusher); ok {
				f.Flush()
			}
		}
	}()

	return nil
}

// sendJSONRPCError sends a JSON-RPC error response
func sendJSONRPCError(ctx fiber.Ctx, id string, code int, message string, data interface{}) error {
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
func WithTaskStore(store task.TaskStore) ManagerOption {
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
