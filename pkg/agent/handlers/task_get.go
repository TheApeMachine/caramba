package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskGetParams defines the expected parameters for the task.get method
// Based on TaskQueryParams in the A2A schema.
type taskGetParams struct {
	TaskID        string `json:"id"`                      // Changed from task_id to id to match TaskQueryParams
	HistoryLength *int   `json:"historyLength,omitempty"` // Pointer to handle optional field
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
			Data:    "Missing required parameter: id", // Changed from task_id
		}
	}

	// Retrieve task from store
	t, err := store.GetTask(getParams.TaskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		// Use A2A specific error code
		return nil, &task.TaskRequestError{
			Code:    -32001, // Task not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", getParams.TaskID, err.Error()),
		}
	}

	// Determine if the task is in a final state
	final := t.Status.State == task.TaskStateCompleted ||
		t.Status.State == task.TaskStateCanceled ||
		t.Status.State == task.TaskStateFailed

	// Prepare metadata, including history if requested
	metadata := make(map[string]any)
	if getParams.HistoryLength != nil && *getParams.HistoryLength > 0 && len(t.History) > 0 {
		historyLen := *getParams.HistoryLength
		if historyLen > len(t.History) {
			historyLen = len(t.History)
		}
		// Get the last N messages
		metadata["history"] = t.History[len(t.History)-historyLen:]
	}

	// Construct the response according to A2A spec (TaskStatusUpdateEvent)
	response := task.TaskStatusUpdateEvent{
		ID:       t.ID,
		Status:   t.Status,
		Final:    final,
		Metadata: metadata,
	}

	return response, nil
}
