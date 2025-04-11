package handlers

import (
	"encoding/json"

	// Needed potentially for history filtering later

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/task"
)

// taskResubscribeParams defines the expected parameters for the tasks/resubscribe method.
// Based on TaskQueryParams in the A2A schema.
type taskResubscribeParams struct {
	TaskID        string         `json:"id"`
	HistoryLength *int           `json:"historyLength,omitempty"` // Pointer to handle optional field
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// TaskResubscribeHandler handles the tasks/resubscribe A2A method.
type TaskResubscribeHandler struct {
	ctx   fiber.Ctx
	store task.TaskStore
	err   error
}

// NewTaskResubscribeHandler creates a new handler for tasks/resubscribe.
func NewTaskResubscribeHandler(
	ctx fiber.Ctx, store task.TaskStore,
) *TaskResubscribeHandler {
	return &TaskResubscribeHandler{
		ctx:   ctx,
		store: store,
		err:   nil,
	}
}

// HandleRequest implements the logic for the tasks/resubscribe A2A method.
// It retrieves the current task status and includes history based on historyLength if specified.
// The response follows the TaskStatusUpdateEvent structure for consistency with other methods.
func (handler *TaskResubscribeHandler) HandleRequest(
	params json.RawMessage,
) (any, *task.TaskRequestError) {
	methodName := "tasks/resubscribe" // Note: Spec uses / notation

	// Parse and validate parameters
	t, resubscribeParams, taskErr := handler.parseParamsAndGetTask(params, methodName)
	if taskErr != nil {
		return nil, taskErr
	}

	// Build and return response
	return handler.buildResponse(t, resubscribeParams), nil
}

// parseParamsAndGetTask handles parameter validation and task retrieval
func (handler *TaskResubscribeHandler) parseParamsAndGetTask(
	params json.RawMessage,
	methodName string,
) (*task.Task, *taskResubscribeParams, *task.TaskRequestError) {
	var resubscribeParams taskResubscribeParams

	if taskErr := parseAndValidateParams(
		params, &resubscribeParams, methodName,
	); taskErr != nil {
		return nil, nil, taskErr
	}

	t, taskErr := getTaskByID(
		handler.store, resubscribeParams.TaskID, methodName,
	)

	if taskErr != nil {
		return nil, nil, taskErr
	}

	return t, &resubscribeParams, nil
}

// buildResponse creates the TaskStatusUpdateEvent response
func (handler *TaskResubscribeHandler) buildResponse(
	t *task.Task,
	resubscribeParams *taskResubscribeParams,
) task.TaskStatusUpdateEvent {
	final := t.Status.State.IsFinal()
	metadata := make(map[string]any)

	if resubscribeParams.HistoryLength != nil && *resubscribeParams.HistoryLength > 0 && len(t.History) > 0 {
		historyForMetadata := getHistorySlice(t.History, resubscribeParams.HistoryLength)
		if len(historyForMetadata) > 0 {
			metadata["history"] = historyForMetadata
		}
	}

	for k, v := range t.Metadata {
		if k != "history" {
			metadata[k] = v
		}
	}

	for k, v := range resubscribeParams.Metadata {
		metadata[k] = v
	}

	return task.TaskStatusUpdateEvent{
		ID:       t.ID,
		Status:   t.Status,
		Final:    final,
		Metadata: metadata,
	}
}
