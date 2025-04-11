package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskGetParams defines the expected parameters for the task.get method
type taskGetParams struct {
	TaskID string `json:"task_id"` // Assuming A2A spec uses task_id
}

func HandleTaskGet(store task.TaskStore, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var getParams taskGetParams
	if err := types.SimdUnmarshalJSON(params, &getParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.get",
			Data:    err.Error(),
		}
	}

	if getParams.TaskID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.get",
			Data:    "Missing required parameter: task_id",
		}
	}

	// Retrieve task from store
	t, err := store.GetTask(getParams.TaskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		return nil, &task.TaskRequestError{
			Code:    -32000, // Example: Application-specific error code for not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", getParams.TaskID, err.Error()),
		}
	}

	// Return task information (e.g., status) as per A2A spec
	// Adjust the response structure based on the spec
	return map[string]interface{}{
		"id":     t.ID,
		"status": t.Status,
	}, nil
}
