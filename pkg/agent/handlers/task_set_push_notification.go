package handlers

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/task"
)

// taskSetPushNotificationParams defines the expected parameters for the task.setPushNotification method
type taskSetPushNotificationParams struct {
	TaskID           string                `json:"task_id"`           // Assuming A2A spec uses task_id
	PushNotification task.PushNotification `json:"push_notification"` // Assuming structure is defined in task package
}

func HandleTaskSetPushNotification(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	methodName := "tasks/pushNotification/set"
	var setParams taskSetPushNotificationParams
	// Use the helper function for parsing
	if taskErr := parseAndValidateParams(params, &setParams, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Use helper to get task and handle not found/missing ID errors
	// Note: A2A spec uses "task_id", but our helper expects "id". We might need to adjust the spec or helper later.
	// For now, let's assume the TaskStore can handle "task_id" if passed, or we adjust the param struct.
	// Let's stick to the struct's "TaskID" for now and pass it to the helper.
	t, taskErr := getTaskByID(store, setParams.TaskID, methodName)
	if taskErr != nil {
		// Check if it's specifically a missing param error to match original logic
		if setParams.TaskID == "" {
			return nil, task.NewMissingParamError(methodName, "task_id") // Report the A2A expected param name
		}
		return nil, taskErr // Return the Task Not Found error from getTaskByID
	}

	// Check agent capabilities using the helper
	if capabilityErr := checkAgentCapability("PushNotifications", methodName); capabilityErr != nil {
		return nil, capabilityErr
	}

	// Store the push notification config in the task's metadata
	if t.Metadata == nil {
		t.Metadata = make(map[string]any)
	}
	t.Metadata["pushNotificationConfig"] = setParams.PushNotification

	// Use helper to update the task in the store
	if taskErr := updateTaskInStore(store, t, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Return null on success as per A2A spec
	var result any = nil
	return result, nil
}
