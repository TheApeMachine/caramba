package task

import "github.com/google/uuid"

type TaskResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      string `json:"id"`
	Result  Task   `json:"result"`
}

func NewTaskResponse(task Task) *TaskResponse {
	return &TaskResponse{
		JSONRPC: "2.0",
		ID:      uuid.New().String(),
		Result:  task,
	}
}
