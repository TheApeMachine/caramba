package handlers

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskGetPushNotificationParams defines the expected parameters for the task.getPushNotification method
type taskGetPushNotificationParams struct {
	TaskID string `json:"task_id"` // Assuming A2A spec uses task_id
}

// HandleTaskGetPushNotification implements the logic for the task.getPushNotification A2A method.
func HandleTaskGetPushNotification(store task.TaskStore, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	var getParams taskGetPushNotificationParams
	if err := types.SimdUnmarshalJSON(params, &getParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.getPushNotification",
			Data:    err.Error(),
		}
	}

	if getParams.TaskID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.getPushNotification",
			Data:    "Missing required parameter: task_id",
		}
	}

	// Retrieve the task
	t, err := store.GetTask(getParams.TaskID)
	if err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32000, // Task not found or store error
			Message: "Task not found or error retrieving task",
			Data:    err.Error(),
		}
	}

	// Retrieve the push notification config from the task's metadata
	config, ok := t.Metadata["pushNotificationConfig"]
	if !ok || config == nil {
		// If no config is found, return an empty object or appropriate error/response
		// according to the A2A spec for when no config is set.
		// Returning an empty map for now.
		return map[string]interface{}{}, nil
	}

	// Return the configuration
	return config, nil
}
