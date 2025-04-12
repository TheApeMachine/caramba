package task

import "github.com/google/uuid"

type TaskResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      string `json:"id"`
	Result  Task   `json:"result"`
}

type TaskResponseOption func(*TaskResponse)

func NewTaskResponse(opts ...TaskResponseOption) *TaskResponse {
	return &TaskResponse{
		JSONRPC: "2.0",
		ID:      uuid.New().String(),
	}
}

func WithResponseTask(result Task) TaskResponseOption {
	return func(task *TaskResponse) {
		task.Result = result
	}
}

func WithResponseError(err error) TaskResponseOption {
	return func(task *TaskResponse) {
		task.Result.AddMessage(
			NewAssistantMessage(err.Error()),
		)
	}
}
