package inmemory

import (
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
)

// Repository is a simple in-memory implementation of task.TaskStore
type Repository struct {
	tasks map[string]*task.Task
	mu    sync.RWMutex
}

func NewRepository() *Repository {
	errnie.Trace("inmemmory.NewRepository")

	return &Repository{
		tasks: make(map[string]*task.Task),
	}
}

// GetTask retrieves a task by its ID.
func (repo *Repository) GetTask(id string) (*task.Task, error) {
	repo.mu.RLock()
	defer repo.mu.RUnlock()

	t, ok := repo.tasks[id]
	if !ok {
		return nil, fmt.Errorf("task not found: %s", id)
	}

	// Return a deep copy to prevent data races
	taskCopy := &task.Task{
		ID:        t.ID,
		SessionID: t.SessionID,
		Status:    t.Status,
		History:   make([]task.Message, len(t.History)),
		Artifacts: make([]task.Artifact, len(t.Artifacts)),
	}

	// Deep copy history
	copy(taskCopy.History, t.History)

	// Deep copy artifacts
	copy(taskCopy.Artifacts, t.Artifacts)

	// Deep copy metadata if it exists
	if t.Metadata != nil {
		taskCopy.Metadata = make(map[string]any)
		for k, v := range t.Metadata {
			taskCopy.Metadata[k] = v
		}
	}

	return taskCopy, nil
}

// UpdateTask updates an existing task.
func (repo *Repository) UpdateTask(t *task.Task) error {
	repo.mu.Lock()
	defer repo.mu.Unlock()

	// Create a deep copy to avoid concurrent modification
	taskCopy := &task.Task{
		ID:        t.ID,
		SessionID: t.SessionID,
		Status:    t.Status,
		History:   make([]task.Message, len(t.History)),
		Artifacts: make([]task.Artifact, len(t.Artifacts)),
	}

	// Deep copy history
	copy(taskCopy.History, t.History)

	// Deep copy artifacts
	copy(taskCopy.Artifacts, t.Artifacts)

	// Deep copy metadata if it exists
	if t.Metadata != nil {
		taskCopy.Metadata = make(map[string]any)
		for k, v := range t.Metadata {
			taskCopy.Metadata[k] = v
		}
	}

	repo.tasks[t.ID] = taskCopy
	return nil
}

// DeleteTask removes a task by its ID.
func (repo *Repository) DeleteTask(id string) error {
	repo.mu.Lock()
	defer repo.mu.Unlock()

	delete(repo.tasks, id)
	return nil
}

// CreateTask saves a new task to the store.
func (repo *Repository) CreateTask(t *task.Task) error {
	// For our simple implementation, CreateTask is the same as UpdateTask
	return repo.UpdateTask(t)
}
