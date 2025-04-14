package task

// TaskStore defines the interface for task persistence
type TaskStore interface {
	// Get retrieves a task by ID
	Get(id string) (*Task, error)

	// Save persists a task
	Save(task *Task) error

	// Update updates an existing task
	Update(task *Task) error

	// Delete removes a task
	Delete(id string) error

	// List retrieves tasks based on a filter
	List(filter *TaskFilter) (*TaskList, error)
}
