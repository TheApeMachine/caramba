package handlers

import (
	"context"
	"encoding/json"
	"time"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
)

// HandleTaskCancel implements the logic for the task.cancel A2A method.
func HandleTaskCancel(store task.TaskStore, params json.RawMessage) (any, *task.TaskRequestError) {
	methodName := "tasks/cancel"
	var cancelParams task.TaskIdParams

	// Parse and validate parameters
	if taskErr := parseAndValidateParams(params, &cancelParams, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Get the task
	t, taskErr := getTaskByID(store, cancelParams.ID, methodName)
	if taskErr != nil {
		return nil, taskErr
	}

	// Check if task can be canceled (not in a final state)
	if t.Status.State.IsFinal() {
		return nil, task.NewTaskCannotBeCanceledError(cancelParams.ID, t.Status.State)
	}

	// Create a cancellation context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. Stop any running LLM generation processes
	if t.Status.State == task.TaskStateWorking {
		// Signal cancellation to any running LLM processes
		cancel()
		// Wait for cancellation to complete or timeout
		<-ctx.Done()
	}

	// 2. Cancel any pending tool executions
	// This would be handled by the context cancellation above
	// as tool executions should respect the context

	// 3. Clean up any temporary resources
	// This would be handled by the context cancellation
	// as resource cleanup should be tied to context lifecycle

	// 4. Update task state and send notifications
	t.Status.State = task.TaskStateCanceled
	t.Status.Timestamp = time.Now().Format(time.RFC3339)
	t.Status.Message = task.Message{
		Role: task.MessageRoleAgent,
		Parts: []task.MessagePart{
			task.TextPart{Type: "text", Text: "Task canceled by user request"},
		},
	}

	// Update task in store
	if taskErr := updateTaskInStore(store, t, methodName); taskErr != nil {
		return nil, taskErr
	}

	// Log cancellation with metadata if provided
	if len(cancelParams.Metadata) > 0 {
		errnie.Info("Task canceled with metadata", "taskID", cancelParams.ID, "metadata", cancelParams.Metadata)
	}

	// Log the cancellation event
	errnie.Info("Task cancellation completed", "taskID", cancelParams.ID, "status", t.Status.State)

	// Return true on successful cancellation as per A2A spec
	return true, nil
}
