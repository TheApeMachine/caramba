package handlers

import (
	"encoding/json"
	"time"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// --- Removed In-Memory Store ---
// var (
// 	taskStore      = make(map[string]*task.Task)
// 	taskStoreMutex sync.RWMutex
// )
// --- End Removed In-Memory Store ---

// taskCreateParams defines the expected parameters for the task.create method
type taskCreateParams struct {
	// Define expected fields based on A2A spec for task.create
	// Example: InputData map[string]interface{} `json:"input_data"`
	// Example: TaskType string `json:"task_type"`
	// For now, assuming no specific creation params are strictly needed by the handler itself
}

func HandleTaskCreate(store task.TaskStore, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var createParams taskCreateParams
	if err := types.SimdUnmarshalJSON(params, &createParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.create",
			Data:    err.Error(),
		}
	}

	// Create a new task
	newTask := task.NewTask() // Assuming NewTask generates ID etc.
	newTask.Status = task.TaskStatus{
		State:     task.TaskStateSubmitted,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	// Store the task using the provided store
	if err := store.CreateTask(newTask); err != nil {
		// Handle potential storage error
		return nil, &task.TaskRequestError{
			Code:    -32000, // Example: Application-specific error code
			Message: "Failed to create task in store",
			Data:    err.Error(),
		}
	}

	// Return the task ID as per A2A spec (usually)
	return map[string]string{"id": newTask.ID}, nil
}
