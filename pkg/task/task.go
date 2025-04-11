package task

import (
	"encoding/json"

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

type TaskState int

const (
	TaskStateSubmitted TaskState = iota
	TaskStateWorking
	TaskStateInputRequired
	TaskStateCompleted
	TaskStateCanceled
	TaskStateFailed
	TaskStateUnknown
)

func (state TaskState) String() string {
	return []string{
		"submitted",
		"working",
		"input-required",
		"completed",
		"canceled",
		"failed",
		"unknown",
	}[state]
}

type TaskRequest struct {
	TaskID           string                 `json:"taskId"`
	Artifact         Artifact               `json:"artifact"`
	PushNotification PushNotification       `json:"pushNotification"`
	Metadata         map[string]interface{} `json:"metadata"`
}

type TaskRequestError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data"`
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
