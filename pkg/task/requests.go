package task

import (
	"github.com/theapemachine/caramba/pkg/jsonrpc"
)

// TaskIdParams represents parameters for task ID-based requests
type TaskIdParams struct {
	ID       string         `json:"id"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// TaskQueryParams represents parameters for task query requests
type TaskQueryParams struct {
	ID            string         `json:"id"`
	HistoryLength *int           `json:"historyLength,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// TaskSendParams represents parameters for sending a task
type TaskSendParams struct {
	ID               string                  `json:"id"`
	SessionID        string                  `json:"sessionId,omitempty"`
	Message          Message                 `json:"message"`
	PushNotification *PushNotificationConfig `json:"pushNotification,omitempty"`
	HistoryLength    *int                    `json:"historyLength,omitempty"`
	Metadata         map[string]any          `json:"metadata,omitempty"`
}

// SendTaskRequest represents a request to send a task
type SendTaskRequest struct {
	jsonrpc.Request
	Params TaskSendParams `json:"params"`
}

// SendTaskResponse represents a response to a send task request
type SendTaskResponse struct {
	jsonrpc.Response
	Result *Task `json:"result,omitempty"`
}

// GetTaskRequest represents a request to get a task
type GetTaskRequest struct {
	jsonrpc.Request
	Params TaskQueryParams `json:"params"`
}

// GetTaskResponse represents a response to a get task request
type GetTaskResponse struct {
	jsonrpc.Response
	Result *Task `json:"result,omitempty"`
}

// CancelTaskRequest represents a request to cancel a task
type CancelTaskRequest struct {
	jsonrpc.Request
	Params TaskIdParams `json:"params"`
}

// CancelTaskResponse represents a response to a cancel task request
type CancelTaskResponse struct {
	jsonrpc.Response
	Result *Task `json:"result,omitempty"`
}

// TaskResubscriptionRequest represents a request to resubscribe to a task
type TaskResubscriptionRequest struct {
	jsonrpc.Request
	Params TaskQueryParams `json:"params"`
}

// SendTaskStreamingRequest represents a request to send a task with streaming
type SendTaskStreamingRequest struct {
	jsonrpc.Request
	Params TaskSendParams `json:"params"`
}

// TaskStatusUpdateEvent represents a task status update event
type TaskStatusUpdateEvent struct {
	ID       string         `json:"id"`
	Status   TaskStatus     `json:"status"`
	Final    bool           `json:"final"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// TaskArtifactUpdateEvent represents a task artifact update event
type TaskArtifactUpdateEvent struct {
	ID       string         `json:"id"`
	Artifact Artifact       `json:"artifact"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// SendTaskStreamingResponse represents a streaming response to a send task request
type SendTaskStreamingResponse struct {
	jsonrpc.Response
	Result any `json:"result,omitempty"` // Can be TaskStatusUpdateEvent or TaskArtifactUpdateEvent
}
