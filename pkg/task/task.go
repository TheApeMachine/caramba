package task

import (
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/service/types"
)

type Task struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history"`
	Artifacts []Artifact     `json:"artifacts"`
	Metadata  map[string]any `json:"metadata"`
}

func NewTask() *Task {
	return &Task{
		ID:        uuid.New().String(),
		SessionID: uuid.New().String(),
	}
}

func NewSessionTask(sessionID string) *Task {
	return &Task{
		ID:        uuid.New().String(),
		SessionID: sessionID,
	}
}

func (task *Task) Unmarshal(data []byte) error {
	return types.SimdUnmarshalJSON(data, task)
}

func (task *Task) Marshal() ([]byte, error) {
	return types.SimdMarshalJSON(task)
}

type TaskStatus struct {
	State     TaskState `json:"state"`
	Message   Message   `json:"message"`
	Timestamp string    `json:"timestamp"`
}

type TaskStatusUpdateEvent struct {
	ID       string         `json:"id"`
	Status   TaskStatus     `json:"status"`
	Final    bool           `json:"final"`
	Metadata map[string]any `json:"metadata"`
}

type TaskArtifactUpdateEvent struct {
	ID       string         `json:"id"`
	Artifact Artifact       `json:"artifact"`
	Metadata map[string]any `json:"metadata"`
}

type TaskSendParams struct {
	ID               string           `json:"id"`
	SessionID        string           `json:"sessionId"`
	Message          Message          `json:"message"`
	HistoryLength    int              `json:"historyLength"`
	PushNotification PushNotification `json:"pushNotification"`
	Metadata         map[string]any   `json:"metadata"`
}

// TaskState represents the various states a task can be in.
type TaskState string

// Constants defining the possible task states according to the A2A specification.
const (
	TaskStateUnknown       TaskState = ""               // Default/unset state
	TaskStateSubmitted     TaskState = "submitted"      // Task has been submitted but not yet processed
	TaskStateWorking       TaskState = "working"        // Task is actively being processed
	TaskStateInputRequired TaskState = "input-required" // Task requires user input
	TaskStateCompleted     TaskState = "completed"      // Task finished successfully
	TaskStateFailed        TaskState = "failed"         // Task execution failed
	TaskStateCanceled      TaskState = "canceled"       // Task was canceled by request
)

// String returns the string representation of the TaskState.
func (s TaskState) String() string {
	return string(s)
}

// IsFinal checks if the task state is a terminal state (completed, failed, or canceled).
func (s TaskState) IsFinal() bool {
	return s == TaskStateCompleted || s == TaskStateFailed || s == TaskStateCanceled
}

type TaskRequest struct {
	TaskID           string           `json:"taskId"`
	Artifact         Artifact         `json:"artifact"`
	PushNotification PushNotification `json:"pushNotification"`
	Metadata         map[string]any   `json:"metadata"`
}

type TaskRequestError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data"`
}

// Error implements the error interface.
func (e *TaskRequestError) Error() string {
	return fmt.Sprintf("A2A Error - Code: %d, Message: %s, Data: %s", e.Code, e.Message, e.Data)
}

func NewTaskRequest(msg json.RawMessage) (*TaskRequest, *TaskRequestError) {
	request := &TaskRequest{}
	return request.Unmarshal(msg)
}

func (request *TaskRequest) Unmarshal(
	params json.RawMessage,
) (*TaskRequest, *TaskRequestError) {
	var reqParams TaskRequest

	if err := json.Unmarshal(params, &reqParams); err != nil {
		return nil, &TaskRequestError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	if reqParams.TaskID == "" {
		return nil, &TaskRequestError{
			Code:    -32602,
			Message: "Invalid params",
			Data:    "task ID is required",
		}
	}

	return request, nil
}

type TaskResponse struct {
	TaskID    string         `json:"taskId"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history"`
	Artifacts []Artifact     `json:"artifacts"`
	Metadata  map[string]any `json:"metadata"`
}

func NewTaskResponse(task *Task) *TaskResponse {
	return &TaskResponse{
		TaskID:    task.ID,
		Status:    task.Status,
		History:   task.History,
		Artifacts: task.Artifacts,
		Metadata:  task.Metadata,
	}
}

type TaskIdParams struct {
	ID       string         `json:"id"`
	Metadata map[string]any `json:"metadata,omitempty"`
}
