package handlers

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/task"
)

// taskGetParams defines the expected parameters for the task.get method
// Based on TaskQueryParams in the A2A schema.
type taskGetParams struct {
	TaskID        string `json:"id"`                      // Changed from task_id to id to match TaskQueryParams
	HistoryLength *int   `json:"historyLength,omitempty"` // Pointer to handle optional field
}

func HandleTaskGet(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	methodName := "tasks/get"
	var getParams taskGetParams
	if taskErr := parseAndValidateParams(params, &getParams, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Retrieve task using helper (handles missing ID and not found)
	t, taskErr := getTaskByID(store, getParams.TaskID, methodName) // TaskID field is already named "id" in struct
	if taskErr != nil {
		return nil, taskErr
	}

	// Determine if the task is in a final state using helper method
	final := t.Status.State.IsFinal()

	// Prepare metadata, including history if requested
	metadata := make(map[string]any)
	if getParams.HistoryLength != nil && *getParams.HistoryLength > 0 && len(t.History) > 0 {
		// Use the helper function to get the potentially truncated history
		historyForMetadata := getHistorySlice(t.History, getParams.HistoryLength)
		if len(historyForMetadata) > 0 { // Only add if there's history after slicing
			metadata["history"] = historyForMetadata
		}
	}

	// Add other metadata from the task if it exists
	for k, v := range t.Metadata {
		if k != "history" { // Avoid overwriting history if it was explicitly requested
			metadata[k] = v
		}
	}

	// Construct the response according to A2A spec (TaskStatusUpdateEvent)
	response := task.TaskStatusUpdateEvent{
		ID:       t.ID,
		Status:   t.Status,
		Final:    final,
		Metadata: metadata, // Use the combined metadata map
	}

	return response, nil
}
