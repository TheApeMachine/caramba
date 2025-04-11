package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
)

// parseAndValidateParams unmarshals the JSON parameters into the target struct
// and returns an InvalidParamsError if unmarshalling fails.
func parseAndValidateParams(params json.RawMessage, target any, methodName string) *task.TaskRequestError {
	if err := types.SimdUnmarshalJSON(params, target); err != nil {
		return task.NewInvalidParamsError(methodName, err)
	}
	return nil
}

// getTaskByID retrieves a task from the store by its ID and handles not found errors.
func getTaskByID(store task.TaskStore, taskID string, methodName string) (*task.Task, *task.TaskRequestError) {
	if taskID == "" {
		return nil, task.NewMissingParamError(methodName, "id") // Ensure consistent param name
	}
	t, err := store.GetTask(taskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		return nil, task.NewTaskNotFoundError(taskID, err)
	}
	return t, nil
}

// updateTaskInStore updates a task in the store and handles internal errors.
func updateTaskInStore(store task.TaskStore, t *task.Task, methodName string) *task.TaskRequestError {
	if err := store.UpdateTask(t); err != nil {
		// Log the error internally
		errnie.Error("Failed to update task in store", "error", err, "taskID", t.ID, "method", methodName)
		// Return a generic internal error to the client
		return task.NewInternalError(methodName, "Failed to update task", err, -32000)
	}
	return nil
}

// getHistorySlice returns a slice of the task history, potentially truncated to the specified length.
func getHistorySlice(history []task.Message, length *int) []task.Message {
	if length != nil && *length > 0 && len(history) > *length {
		startIndex := len(history) - *length
		return history[startIndex:]
	}
	// Return the full history if length is nil, zero, or greater than actual length
	return history
}

// checkAgentCapability checks if the agent has the required capability.
// TODO: Refine signature/implementation based on how capabilities are accessed (e.g., pass a capabilities map/struct).
func checkAgentCapability(capability string, methodName string) *task.TaskRequestError {
	// Placeholder implementation - assumes capabilities are not available here
	// In a real scenario, this would check against the agent's actual capabilities
	hasCapability := false // Default to false for safety

	if !hasCapability {
		// Use a specific error type if available, otherwise use a generic one or define NewCapabilityError
		// return task.NewCapabilityError(methodName, capability)
		return task.NewInternalError(methodName, fmt.Sprintf("Agent lacks required capability: %s", capability), nil, -32002) // Example error code
	}
	return nil
}
