package handlers

import (
	"encoding/json"
	"fmt"

	// Needed potentially for history filtering later
	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskResubscribeParams defines the expected parameters for the tasks/resubscribe method.
// Based on TaskQueryParams in the A2A schema.
type taskResubscribeParams struct {
	TaskID        string         `json:"id"`
	HistoryLength *int           `json:"historyLength,omitempty"` // Pointer to handle optional field
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// HandleTaskResubscribe implements the logic for the tasks/resubscribe A2A method.
// Currently, it retrieves and returns the current task status.
// TODO: Implement history retrieval based on historyLength.
func HandleTaskResubscribe(store task.TaskStore, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var resubscribeParams taskResubscribeParams
	if err := types.SimdUnmarshalJSON(params, &resubscribeParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for tasks/resubscribe",
			Data:    err.Error(),
		}
	}

	if resubscribeParams.TaskID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for tasks/resubscribe",
			Data:    "Missing required parameter: id",
		}
	}

	// Retrieve task from store
	t, err := store.GetTask(resubscribeParams.TaskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		// Use A2A specific error code for Task Not Found
		return nil, &task.TaskRequestError{
			Code:    -32001, // Task not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", resubscribeParams.TaskID, err.Error()),
		}
	}

	// TODO: If resubscribeParams.HistoryLength is not nil, filter/include history in response.
	// For now, just return the status.

	// Return task status as per A2A spec (assuming TaskStatus is appropriate)
	// Adjust the response structure if the spec implies something else for resubscribe.
	return t.Status, nil
}
