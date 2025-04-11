package handlers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
)

// HandleTaskSendSubscribe handles the tasks/sendSubscribe method, streaming SSE results directly.
func HandleTaskSendSubscribe(
	ctx fiber.Ctx,
	store task.TaskStore,
	llmProvider provider.ProviderType,
	toolRegistry *tools.Registry,
	params json.RawMessage,
	requestID interface{},
) error {
	var sendParams taskSendParams

	if err := types.SimdUnmarshalJSON(params, &sendParams); err != nil {
		return &task.TaskRequestError{
			Code:    -32602,
			Message: "Invalid params for tasks/sendSubscribe",
			Data:    err.Error(),
		}
	}

	if sendParams.ID == "" {
		return &task.TaskRequestError{
			Code:    -32602,
			Message: "Invalid params for tasks/sendSubscribe",
			Data:    "Missing required parameter: id",
		}
	}

	// --- Task Retrieval & Initial Update ---
	t, taskErr := retrieveAndPrepareTask(store, sendParams) // Reuse helper
	if taskErr != nil {
		return taskErr // Return error to be formatted as JSONRPCError by the caller
	}

	// --- Prepare LLM Call Parameters ---
	llmParams, err := prepareLLMParams(t, toolRegistry, sendParams.HistoryLength) // Reuse helper
	if err != nil {
		// Attempt to update task state before returning error
		updateTaskToFailed(store, t, "Failed to prepare LLM parameters", err) // Reuse helper
		return &task.TaskRequestError{Code: -32000, Message: "Failed to prepare LLM parameters", Data: err.Error()}
	}

	// --- Initiate Streaming Generation ---
	providerEventChan, err := llmProvider.Generate(llmParams)
	if err != nil {
		errnie.Error(err, "taskID", t.ID)
		updateTaskToFailed(store, t, "Failed to initiate LLM generation", err)                                       // Reuse helper
		return &task.TaskRequestError{Code: -32003, Message: "Failed to initiate LLM generation", Data: err.Error()} // Use appropriate code?
	}

	// --- Set SSE Headers ---
	ctx.Set("Content-Type", "text/event-stream")
	ctx.Set("Cache-Control", "no-cache")
	ctx.Set("Connection", "keep-alive")
	// ctx.Set("Transfer-Encoding", "chunked") // Fiber handles chunked encoding with SendStreamWriter

	// --- Use SendStreamWriter for robust streaming ---
	err = ctx.SendStreamWriter(func(w *bufio.Writer) { // Correct signature: no error return
		errnie.Info("Starting SSE stream writer for tasks/sendSubscribe", "taskID", t.ID, "requestID", requestID)
		defer errnie.Info("Exiting SSE stream writer for tasks/sendSubscribe", "taskID", t.ID, "requestID", requestID)

		var accumulatedContent strings.Builder
		var accumulatedToolCalls []provider.PendingToolCall
		var assistantMessage provider.Message
		var streamErr error

		// --- Send Initial Working Status ---
		// We send the initial 'working' status update immediately after starting stream
		initialStatusEvent := task.TaskStatusUpdateEvent{
			ID: t.ID,
			Status: task.TaskStatus{
				State:     task.TaskStateWorking,
				Timestamp: t.Status.Timestamp, // Use timestamp from retrieveAndPrepareTask
				// Message:   t.Status.Message, // Optionally include the initial user message?
			},
			Final: false,
		}
		if err := writeSSEEvent(w, requestID, initialStatusEvent); err != nil {
			errnie.New(errnie.WithError(err))
			updateTaskToFailed(store, t, "Failed to write initial SSE status", err)
			return
		}

		isStreaming := true

		for isStreaming {
			for event := range providerEventChan {
				errnie.Debug("Received provider event in sendSubscribe", "taskID", t.ID)

				if event.Message.Role == "error" {
					errnie.Error("Received error event from provider", "taskID", t.ID, "error", event.Message.Content)
					streamErr = fmt.Errorf("provider stream error: %s", event.Message.Content)
					isStreaming = false
					break
				}

				// Store assistant message if it contains tool calls
				if event.Message.Role == "assistant" && len(event.ToolCalls) > 0 {
					assistantMessage = event.Message
				}

				// Handle Message Content -> TaskArtifactUpdateEvent
				if event.Message.Content != "" {
					accumulatedContent.WriteString(event.Message.Content)
					artifactUpdate := task.TaskArtifactUpdateEvent{
						ID: t.ID,
						Artifact: task.Artifact{
							Index:     0,    // Assuming one artifact stream for now
							Append:    true, // Append content chunks
							LastChunk: false,
							Parts:     []task.Part{{Text: event.Message.Content}}, // Assign struct value directly
						},
					}
					if err := writeSSEEvent(w, requestID, artifactUpdate); err != nil {
						errnie.Error("Failed to write SSE artifact chunk", "error", err, "taskID", t.ID)
						streamErr = err // Store error and break
						isStreaming = false
						break
					}
				}

				// Accumulate Tool Calls
				if len(event.ToolCalls) > 0 {
					errnie.Info("Received tool call request(s) in sendSubscribe stream", "taskID", t.ID, "count", len(event.ToolCalls))
					accumulatedToolCalls = append(accumulatedToolCalls, event.ToolCalls...)
				}
			} // End of providerEventChan loop

			// Synthesize assistant message if needed
			if assistantMessage.Role == "" && len(accumulatedToolCalls) > 0 {
				assistantMessage = provider.Message{Role: "assistant", Content: accumulatedContent.String()}
			}

			// Add assistant's streamed response/request to history *before* tool execution
			addAssistantStreamResponseToHistory(t, streamProcessingResult{
				accumulatedContent:   accumulatedContent.String(),
				accumulatedToolCalls: accumulatedToolCalls,
				assistantMessage:     assistantMessage,
				err:                  nil, // Error handled separately below
			})
		}

		// --- Handle End of Stream / Errors / Tool Execution ---
		finalAgentMessage := task.Message{}
		finalTaskState := task.TaskStateFailed // Default to failed unless successful completion

		if streamErr != nil {
			// Error during stream processing from provider
			errnie.Error("Error processing provider stream", "error", streamErr, "taskID", t.ID)
			updateTaskToFailed(store, t, "Provider stream error", streamErr)
			finalAgentMessage = task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Text: "ERROR: " + streamErr.Error()}}}
			finalTaskState = task.TaskStateFailed

		} else if len(accumulatedToolCalls) > 0 {
			// Execute tools and get final response (non-streaming for the final bit)
			finalAgentMessage, finalTaskState = executeToolcall(
				finalAgentMessage,
				finalTaskState,
				t,
				accumulatedToolCalls,
				toolRegistry,
				llmProvider,
				llmParams,
				store,
				w,
				requestID,
			)
		} else {
			// No tool calls, stream finished successfully
			finalTaskState, finalAgentMessage = streamFinished(
				finalTaskState,
				finalAgentMessage,
				accumulatedContent,
				t,
				w,
				requestID,
				store,
			)
		}

		// --- Update Task and Send Final Status ---
		// Update the task in the store *before* sending the final SSE status
		updateFinalTaskStatus(store, t, finalAgentMessage, finalTaskState) // Reuse helper

		// Send final status update via SSE
		finalStatusEvent := task.TaskStatusUpdateEvent{
			ID:     t.ID,
			Status: t.Status, // Use the status updated by updateFinalTaskStatus
			Final:  true,
		}
		if err := writeSSEEvent(w, requestID, finalStatusEvent); err != nil {
			// Log error, but can't do much else as stream is ending
			errnie.Error("Failed to write final SSE status", "error", err, "taskID", t.ID)
		}

		// Log completion
		errnie.Info("SSE stream completed for tasks/sendSubscribe", "taskID", t.ID, "finalState", t.Status.State)
	}) // End of SendStreamWriter func

	// Check if SendStreamWriter itself returned an error (e.g., connection closed prematurely)
	if err != nil {
		errnie.Error("SendStreamWriter failed", "error", err, "taskID", t.ID)
		// Task status might be inconsistent if error happened after some events were sent.
		// Update task to failed if it wasn't already.
		currentTask, _ := store.GetTask(t.ID) // Fetch latest status
		if currentTask != nil && currentTask.Status.State != task.TaskStateFailed && currentTask.Status.State != task.TaskStateCanceled {
			updateTaskToFailed(store, t, "SSE stream writer failed", err)
		}
		// Cannot return a JSONRPC error here as headers were likely sent.
	}

	// If setup succeeded, we return nil, as the response was handled by SendStreamWriter.
	return nil
}

func streamFinished(
	finalTaskState task.TaskState,
	finalAgentMessage task.Message,
	accumulatedContent strings.Builder,
	t *task.Task,
	w *bufio.Writer,
	requestID interface{},
	store task.TaskStore,
) (task.TaskState, task.Message) {
	finalTaskState = task.TaskStateCompleted
	finalAgentMessage = task.Message{
		Role: task.MessageRoleAgent,
		Parts: []task.MessagePart{
			task.TextPart{Text: accumulatedContent.String()},
		},
	}

	// Convert finalAgentMessage.Parts ([]MessagePart) to []Part
	finalParts := make([]task.Part, len(finalAgentMessage.Parts))
	for i, p := range finalAgentMessage.Parts {
		finalParts[i] = task.Part{Text: fmt.Sprintf("[Unsupported Part: %T]", p)} // Assign fallback struct value
	}

	// Send final accumulated content as the last artifact chunk
	finalArtifactUpdate := task.TaskArtifactUpdateEvent{
		ID: t.ID,
		Artifact: task.Artifact{
			Index:     0,
			Append:    false,
			LastChunk: true,
			Parts:     finalParts, // Use the converted slice
		},
	}
	if err := writeSSEEvent(w, requestID, finalArtifactUpdate); err != nil {
		errnie.Error("Failed to write final SSE artifact", "error", err, "taskID", t.ID)
		updateTaskToFailed(store, t, "Failed to write final artifact SSE", err)
		finalTaskState = task.TaskStateFailed // Mark failed if write fails
	}
	return finalTaskState, finalAgentMessage
}

func executeToolcall(finalAgentMessage task.Message, finalTaskState task.TaskState, t *task.Task, accumulatedToolCalls []provider.PendingToolCall, toolRegistry *tools.Registry, llmProvider provider.ProviderType, llmParams provider.ProviderParams, store task.TaskStore, w *bufio.Writer, requestID interface{}) (task.Message, task.TaskState) {
	var toolExecErr error
	finalAgentMessage, finalTaskState, toolExecErr = executeToolsAndGetResponse( // Reuse helper
		t, accumulatedToolCalls, toolRegistry, llmProvider, llmParams, store,
	)
	if toolExecErr != nil {
		// Error message and state are already set by executeToolsAndGetResponse
		errnie.Warn("Tool execution or final response failed in sendSubscribe", "taskID", t.ID, "error", toolExecErr)
		// The task state should already be Failed in t.Status if error occurred
	} else {
		// Successful tool execution and final response
		finalTaskState = task.TaskStateCompleted

		// Convert finalAgentMessage.Parts ([]MessagePart) to []Part
		finalParts := make([]task.Part, len(finalAgentMessage.Parts))
		for i, p := range finalAgentMessage.Parts {
			finalParts[i] = task.Part{Text: fmt.Sprintf("[Unsupported Part: %T]", p)} // Assign fallback struct value
		}

		// Send the final message as an artifact update
		finalArtifactUpdate := task.TaskArtifactUpdateEvent{
			ID: t.ID,
			Artifact: task.Artifact{
				Index:     0,
				Append:    false,
				LastChunk: true,
				Parts:     finalParts, // Use the converted slice
			},
		}
		if err := writeSSEEvent(w, requestID, finalArtifactUpdate); err != nil {
			errnie.Error("Failed to write final SSE artifact after tools", "error", err, "taskID", t.ID)
			// Update status to failed even if tools worked but write failed
			updateTaskToFailed(store, t, "Failed to write final artifact SSE", err)
			finalTaskState = task.TaskStateFailed
		}
	}
	return finalAgentMessage, finalTaskState
}

// writeSSEEvent formats and writes a JSON-RPC structured SSE event.
func writeSSEEvent(w *bufio.Writer, requestID interface{}, payload interface{}) error {
	// Wrap the payload (TaskStatusUpdateEvent or TaskArtifactUpdateEvent)
	// in the JSON-RPC response structure.
	sseResponse := types.JSONRPCResponse{
		Version: "2.0",
		ID:      requestID, // Use the original request ID
		Result:  payload,   // The actual event data goes here
	}

	jsonData, err := types.SimdMarshalJSON(sseResponse)
	if err != nil {
		return fmt.Errorf("failed to marshal SSE event payload: %w", err)
	}

	// Write the SSE formatted message ("data: {json}\n\n")
	// log.Printf("Writing SSE Data: data: %s\n\n", string(jsonData)) // Debug logging
	_, err = fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
	if err != nil {
		return fmt.Errorf("failed to write SSE event data: %w", err)
	}

	// Flush the data to the client
	if err := w.Flush(); err != nil {
		return fmt.Errorf("failed to flush SSE event data: %w", err)
	}

	return nil
}
