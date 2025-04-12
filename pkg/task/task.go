package task

import (
	"fmt"

	"github.com/google/uuid"
)

type Task struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history"`
	Artifacts []Artifact     `json:"artifacts"`
	Metadata  map[string]any `json:"metadata"`
}

type TaskOption func(*Task)

func NewTask(opts ...TaskOption) Task {
	task := &Task{
		ID:        uuid.New().String(),
		SessionID: uuid.New().String(),
		Status: TaskStatus{
			State: TaskStateSubmitted,
		},
		History:   make([]Message, 0),
		Artifacts: make([]Artifact, 0),
		Metadata:  make(map[string]any),
	}

	for _, opt := range opts {
		opt(task)
	}

	return *task
}

func (task *Task) AddResult(result *TaskResponse) {
	task.History = result.Result.History
	task.Artifacts = result.Result.Artifacts
	task.Metadata = result.Result.Metadata
}

func (task *Task) AddMessage(message Message) {
	task.History = append(task.History, message)
}

type TaskStatus struct {
	State TaskState `json:"state"`
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
	JSONRPC string `json:"jsonrpc"`
	ID      string `json:"id"`
	Method  string `json:"method"`
	Params  Task   `json:"params"`
}

func NewTaskRequest(task Task) *TaskRequest {
	return &TaskRequest{
		JSONRPC: "2.0",
		ID:      task.ID,
		Method:  "tasks/send",
		Params:  task,
	}
}

func (tr *TaskRequest) AddMessage(message Message) {
	tr.Params.AddMessage(message)
}

func (tr *TaskRequest) AddResult(result *TaskResponse) {
	tr.Params.AddResult(result)
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

type TaskResult struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId"`
	Status    TaskStatus     `json:"status"`
	Artifacts []Artifact     `json:"artifacts"`
	Metadata  map[string]any `json:"metadata"`
}

type TaskIdParams struct {
	ID       string         `json:"id"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func WithMessages(messages ...Message) TaskOption {
	return func(task *Task) {
		task.History = append(task.History, messages...)
	}
}
