package handlers

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/task"
)

// taskCancelParams defines the expected parameters for the task.cancel method
// Based on TaskIdParams in the A2A schema.
type taskCancelParams struct {
	ID       string         `json:"id"` // Renamed from TaskID
	Metadata map[string]any `json:"metadata,omitempty"`
}

// HandleTaskCancel implements the logic for the task.cancel A2A method.
func HandleTaskCancel(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	methodName := "tasks/cancel"
	var cancelParams task.TaskIdParams

	// Parse and validate parameters
	if taskErr := parseAndValidateParams(params, &cancelParams, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Get the task
	t, taskErr := getTaskByID(store, cancelParams.ID, methodName)
	if taskErr != nil {
		return nil, taskErr
	}

	// Check if task can be canceled (not in a final state)
	if t.Status.State.IsFinal() {
		return nil, task.NewTaskCannotBeCanceledError(cancelParams.ID, t.Status.State)
	}

	// Update task state to canceled
	t.Status.State = task.TaskStateCanceled
	t.Status.Timestamp = time.Now().Format(time.RFC3339)
	t.Status.Message = task.Message{
		Role: task.MessageRoleAgent,
		Parts: []task.MessagePart{
			task.TextPart{Type: "text", Text: "Task canceled by user request"},
		},
	}

	// Update task in store
	if taskErr := updateTaskInStore(store, t, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Log if metadata was provided (optional)
	if len(cancelParams.Metadata) > 0 {
		// Consider using errnie for logging if available/preferred
		fmt.Printf("INFO: %s called for task %s with metadata: %v\n", methodName, cancelParams.ID, cancelParams.Metadata)
	}

	// TODO: Perform actual cancellation logic (e.g., stop background processes)
	fmt.Printf("TODO: Perform cancellation for task %s\n", cancelParams.ID)

	// TODO: Send SSE update via the A2A service instance if needed
	// Example: srv.SendTaskUpdate(cancelParams.ID, map[string]any{ "status": t.Status })

	// Return true on successful cancellation as per A2A spec
	return true, nil
}
