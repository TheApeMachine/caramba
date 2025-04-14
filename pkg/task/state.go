package task

import "errors"

// TaskState represents the possible states of a task in the A2A protocol
type TaskState string

const (
	TaskStateSubmitted     TaskState = "submitted"
	TaskStateWorking       TaskState = "working"
	TaskStateInputRequired TaskState = "input-required"
	TaskStateCompleted     TaskState = "completed"
	TaskStateCanceled      TaskState = "canceled"
	TaskStateFailed        TaskState = "failed"
	TaskStateUnknown       TaskState = "unknown"
)

// Validate checks if the task state is valid
func (s TaskState) Validate() error {
	switch s {
	case TaskStateSubmitted, TaskStateWorking, TaskStateInputRequired,
		TaskStateCompleted, TaskStateCanceled, TaskStateFailed, TaskStateUnknown:
		return nil
	default:
		return errors.New("invalid task state")
	}
}

// String returns the string representation of the task state
func (s TaskState) String() string {
	return string(s)
}

// IsFinal returns true if the task state is a final state
func (s TaskState) IsFinal() bool {
	return s == TaskStateCompleted || s == TaskStateFailed || s == TaskStateCanceled
}
