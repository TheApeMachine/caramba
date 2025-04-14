package task

import (
	"github.com/theapemachine/caramba/pkg/errors"
	"github.com/theapemachine/caramba/pkg/jsonrpc"
)

// TaskResponse represents a JSON-RPC response for task operations
type TaskResponse struct {
	jsonrpc.Response
	Result *Task `json:"result,omitempty"`
}

// TaskResponseOption represents a task response configuration option
type TaskResponseOption func(*TaskResponse)

// NewTaskResponse creates a new task response with optional configuration
func NewTaskResponse(opts ...TaskResponseOption) *TaskResponse {
	response := &TaskResponse{}

	for _, opt := range opts {
		opt(response)
	}

	return response
}

// WithResponseID sets the response ID
func WithResponseID(id string) TaskResponseOption {
	return func(r *TaskResponse) {
		r.ID = id
	}
}

// WithResponseTask sets the response task
func WithResponseTask(task Task) TaskResponseOption {
	return func(r *TaskResponse) {
		r.Result = &task
	}
}

// WithResponseError sets the response error
func WithResponseError(err error) TaskResponseOption {
	return func(r *TaskResponse) {
		if err != nil {
			switch e := err.(type) {
			case *errors.JSONRPCError:
				r.Error = e
			default:
				r.Error = &errors.JSONRPCError{
					Code:    -32603, // Internal error
					Message: err.Error(),
				}
			}
		}
	}
}
