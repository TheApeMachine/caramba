package handlers

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/task"
)

// taskGetPushNotificationParams defines the expected parameters for the task.getPushNotification method
type taskGetPushNotificationParams struct {
	TaskID string `json:"task_id"` // Assuming A2A spec uses task_id
}

// HandleTaskGetPushNotification implements the logic for the task.getPushNotification A2A method.
func HandleTaskGetPushNotification(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	methodName := "tasks/pushNotification/get"
	var getParams taskGetPushNotificationParams
	if taskErr := parseAndValidateParams(params, &getParams, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Retrieve the task using helper (handles missing ID and not found)
	// Similar issue as setPushNotification with task_id vs id. Passing TaskID from struct.
	t, taskErr := getTaskByID(store, getParams.TaskID, methodName)
	if taskErr != nil {
		// Check if it's specifically a missing param error to match original logic
		if getParams.TaskID == "" {
			return nil, task.NewMissingParamError(methodName, "task_id") // Report the A2A expected param name
		}
		return nil, taskErr // Return the Task Not Found error from getTaskByID
	}

	// Check agent capabilities using the helper
	if capabilityErr := checkAgentCapability("PushNotifications", methodName); capabilityErr != nil {
		return nil, capabilityErr
	}

	// Retrieve the push notification config from the task's metadata
	config, ok := t.Metadata["pushNotificationConfig"]
	if !ok || config == nil {
		// If no config is found, return null as per A2A spec
		var result any = nil
		return result, nil
	}

	// Return the configuration
	return config, nil
}
