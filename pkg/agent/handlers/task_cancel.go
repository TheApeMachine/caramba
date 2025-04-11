package handlers

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskCancelParams defines the expected parameters for the task.cancel method
// Based on TaskIdParams in the A2A schema.
type taskCancelParams struct {
	ID       string         `json:"id"` // Renamed from TaskID
	Metadata map[string]any `json:"metadata,omitempty"`
}

// HandleTaskCancel implements the logic for the task.cancel A2A method.
func HandleTaskCancel(store task.TaskStore, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var cancelParams taskCancelParams
	if err := types.SimdUnmarshalJSON(params, &cancelParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.cancel",
			Data:    err.Error(),
		}
	}

	if cancelParams.ID == "" { // Check ID
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.cancel",
			Data:    "Missing required parameter: id", // Use id
		}
	}

	// Retrieve task from store
	t, err := store.GetTask(cancelParams.ID) // Use ID
	if err != nil {
		// Handle potential store errors (e.g., not found)
		// Use A2A specific error code
		return nil, &task.TaskRequestError{
			Code:    -32001, // Task not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", cancelParams.ID, err.Error()),
		}
	}

	// Check if task is already in a final state (cannot be cancelled)
	if t.Status.State == task.TaskStateCompleted || t.Status.State == task.TaskStateCanceled || t.Status.State == task.TaskStateFailed {
		return nil, &task.TaskRequestError{
			Code:    -32002, // Task cannot be cancelled
			Message: "Task cannot be canceled",
			Data:    fmt.Sprintf("Task '%s' is already in a final state ('%s')", cancelParams.ID, t.Status.State.String()), // Assuming TaskState has String() method
		}
	}

	// Update task status to Canceled
	t.Status = task.TaskStatus{
		State:     task.TaskStateCanceled,
		Timestamp: time.Now().Format(time.RFC3339),
		// Optionally add a cancellation message
		// Message: task.Message{ Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: "Task canceled by request"}} },
	}

	// Update the task in the store
	if err := store.UpdateTask(t); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32000, // Example: Application-specific error code (Could refine later)
			Message: "Failed to update task status to canceled in store",
			Data:    err.Error(),
		}
	}

	// Log if metadata was provided (optional)
	if len(cancelParams.Metadata) > 0 {
		fmt.Printf("INFO: task.cancel called for task %s with metadata: %v\n", cancelParams.ID, cancelParams.Metadata)
	}

	// TODO: Perform actual cancellation logic (e.g., stop background processes)
	fmt.Printf("TODO: Perform cancellation for task %s\n", cancelParams.ID)

	// TODO: Send SSE update via the A2A service instance if needed
	// Example: srv.SendTaskUpdate(cancelParams.ID, map[string]interface{}{"status": t.Status})

	// Return true on successful cancellation as per A2A spec
	return true, nil
}
