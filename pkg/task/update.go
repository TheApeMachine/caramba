package task

// TaskUpdate represents an update to a task
type TaskUpdate struct {
	Name    *string    `json:"name,omitempty"`
	State   *TaskState `json:"state,omitempty"`
	Message *string    `json:"message,omitempty"`
}
