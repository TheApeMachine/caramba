package handlers

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskCancelParams defines the expected parameters for the task.cancel method
type taskCancelParams struct {
	TaskID string `json:"task_id"` // Assuming A2A spec uses task_id
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

	if cancelParams.TaskID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.cancel",
			Data:    "Missing required parameter: task_id",
		}
	}

	// Retrieve task from store
	t, err := store.GetTask(cancelParams.TaskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		return nil, &task.TaskRequestError{
			Code:    -32000, // Example: Application-specific error code for not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", cancelParams.TaskID, err.Error()),
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
			Code:    -32000, // Example: Application-specific error code
			Message: "Failed to update task status to canceled in store",
			Data:    err.Error(),
		}
	}

	// TODO: Perform actual cancellation logic (e.g., stop background processes)
	fmt.Printf("TODO: Perform cancellation for task %s\n", cancelParams.TaskID)

	// TODO: Send SSE update via the A2A service instance if needed
	// Example: srv.SendTaskUpdate(cancelParams.TaskID, map[string]interface{}{"status": t.Status})

	// Return the updated status as confirmation
	return map[string]interface{}{
		"status": t.Status,
	}, nil
}
