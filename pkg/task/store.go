package task

// TaskStore defines the interface for persisting and retrieving tasks.
type TaskStore interface {
	// CreateTask saves a new task.
	CreateTask(task *Task) error
	// GetTask retrieves a task by its ID.
	GetTask(taskID string) (*Task, error)
	// UpdateTask updates an existing task.
	UpdateTask(task *Task) error
	// DeleteTask removes a task by its ID (optional, implement if needed).
	// DeleteTask(taskID string) error
}
