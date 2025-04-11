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
			Code:    -32001,           // Task not found
			Message: "Task not found", // Simplified message
			Data:    err.Error(),
		}
	}

	// TODO: Check agent capabilities here. If !srv.card.Capabilities.PushNotifications, return error -32003

	// Retrieve the push notification config from the task's metadata
	config, ok := t.Metadata["pushNotificationConfig"]
	if !ok || config == nil {
		// If no config is found, return null as per A2A spec
		var result interface{} = nil
		return result, nil
	}

	// Return the configuration
	return config, nil
}
