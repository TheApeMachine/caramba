package task

import (
	"encoding/json"
	"io"
	"time"

	"github.com/google/uuid"
)

// TaskStatus represents the current status of a task
type TaskStatus struct {
	State     TaskState `json:"state"`
	Message   *Message  `json:"message,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// Task represents a task in the A2A protocol
type Task struct {
	ID        string         `json:"id"`
	SessionID *string        `json:"sessionId,omitempty"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history,omitempty"`
	Artifacts []Artifact     `json:"artifacts,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// NewTask creates a new task with optional configuration
func NewTask(opts ...TaskOption) *Task {
	task := &Task{
		ID:        uuid.New().String(),
		Status:    TaskStatus{State: TaskStateSubmitted},
		History:   make([]Message, 0),
		Metadata:  make(map[string]any),
		Artifacts: make([]Artifact, 0),
	}

	for _, opt := range opts {
		opt(task)
	}

	return task
}

func (task *Task) Read(p []byte) (n int, err error) {
	data, err := json.Marshal(task)

	if err != nil {
		return 0, err
	}

	copy(p, data)
	return len(data), io.EOF
}

func (task *Task) Write(p []byte) (n int, err error) {
	if err := json.Unmarshal(p, task); err != nil {
		return 0, err
	}

	return len(p), nil
}

// TaskOption represents a task configuration option
type TaskOption func(*Task)

// WithMessages adds initial messages to the task
func WithMessages(messages ...Message) TaskOption {
	return func(t *Task) {
		t.History = append(t.History, messages...)
	}
}

// WithMetadata adds metadata to the task
func WithMetadata(metadata map[string]any) TaskOption {
	return func(t *Task) {
		t.Metadata = metadata
	}
}

// WithSessionID sets the session ID for the task
func WithSessionID(sessionID string) TaskOption {
	return func(t *Task) {
		t.SessionID = &sessionID
	}
}

// AddMessage adds a message to the task history
func (t *Task) AddMessage(msg Message) {
	t.History = append(t.History, msg)
}

// TaskFilter represents the filter criteria for querying tasks
type TaskFilter struct {
	ID          *string    `json:"id,omitempty"`
	Name        *string    `json:"name,omitempty"`
	State       *TaskState `json:"state,omitempty"`
	CreatedFrom *time.Time `json:"createdFrom,omitempty"`
	CreatedTo   *time.Time `json:"createdTo,omitempty"`
	UpdatedFrom *time.Time `json:"updatedFrom,omitempty"`
	UpdatedTo   *time.Time `json:"updatedTo,omitempty"`
}

// TaskList represents a paginated list of tasks
type TaskList struct {
	Items      []Task `json:"items"`
	TotalCount int64  `json:"totalCount"`
	HasMore    bool   `json:"hasMore"`
}
