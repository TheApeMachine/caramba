package task

import (
	"github.com/theapemachine/caramba/pkg/jsonrpc"
)

// TaskRequest represents a JSON-RPC request for task operations
type TaskRequest struct {
	jsonrpc.Request
	Params *Task `json:"params"`
}

// NewTaskRequest creates a new task request
func NewTaskRequest(task *Task) *TaskRequest {
	return &TaskRequest{
		Request: jsonrpc.NewRequest("tasks/send", task),
		Params:  task,
	}
}
