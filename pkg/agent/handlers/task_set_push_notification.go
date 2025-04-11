package handlers

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskSetPushNotificationParams defines the expected parameters for the task.setPushNotification method
type taskSetPushNotificationParams struct {
	TaskID           string                `json:"task_id"`           // Assuming A2A spec uses task_id
	PushNotification task.PushNotification `json:"push_notification"` // Assuming structure is defined in task package
}

func HandleTaskSetPushNotification(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	var setParams taskSetPushNotificationParams
	if err := types.SimdUnmarshalJSON(params, &setParams); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.setPushNotification",
			Data:    err.Error(),
		}
	}

	if setParams.TaskID == "" {
		return nil, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.setPushNotification",
			Data:    "Missing required parameter: task_id",
		}
	}

	// Retrieve the task
	t, err := store.GetTask(setParams.TaskID)
	if err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32001,           // Task not found
			Message: "Task not found", // Simplified message
			Data:    err.Error(),
		}
	}

	// TODO: Check agent capabilities here. If !srv.card.Capabilities.PushNotifications, return error -32003

	// Store the push notification config in the task's metadata
	if t.Metadata == nil {
		t.Metadata = make(map[string]any)
	}
	t.Metadata["pushNotificationConfig"] = setParams.PushNotification

	// Update the task in the store
	if err := store.UpdateTask(t); err != nil {
		return nil, &task.TaskRequestError{
			Code:    -32603, // Internal error
			Message: "Internal error: Failed to update task with push notification config",
			Data:    err.Error(),
		}
	}

	// Return null on success as per A2A spec
	var result interface{} = nil
	return result, nil
}
