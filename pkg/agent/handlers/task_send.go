package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
)

// StreamUpdateSender defines the interface required to send updates.
// Duplicated here to avoid import cycle. Consider moving to a shared package.
type StreamUpdateSender interface {
	SendTaskUpdate(taskID string, update any)
}

// streamProcessingResult holds the accumulated data from the LLM stream.
type streamProcessingResult struct {
	accumulatedContent   string
	accumulatedToolCalls []provider.PendingToolCall
	assistantMessage     provider.Message // Store the assistant's message that led to tool calls
	err                  error
}

// taskSendParams defines the expected parameters for the task.send method.
// Based on TaskSendParams in the A2A schema.
type taskSendParams struct {
	ID               string                 `json:"id"` // Renamed from TaskID
	SessionID        string                 `json:"sessionId,omitempty"`
	Message          task.Message           `json:"message"`
	PushNotification *task.PushNotification `json:"pushNotification,omitempty"`
	HistoryLength    *int                   `json:"historyLength,omitempty"`
	Metadata         map[string]any         `json:"metadata,omitempty"`
}

// HandleTaskSend orchestrates the streaming task request.
func HandleTaskSend(
	store task.TaskStore,
	llmProvider provider.ProviderType,
	toolRegistry *tools.Registry,
	updater StreamUpdateSender, // Use the interface
	params json.RawMessage,
) (any, *task.TaskRequestError) {

	// --- Parameter Parsing ---
	methodName := "tasks/send"
	var sendParams taskSendParams
	// Use the helper function for parsing
	if taskErr := parseAndValidateParams(params, &sendParams, methodName); taskErr != nil {
		return nil, taskErr // Returns InvalidParamsError
	}

	// Explicitly check for missing ID after parsing
	if sendParams.ID == "" { // Check ID instead of TaskID
		return nil, task.NewMissingParamError(methodName, "id")
	}

	// --- Task Retrieval & Initial Update ---
	// Pass the full sendParams to potentially use PushNotification config
	t, taskErr := retrieveAndPrepareTask(store, sendParams)
	if taskErr != nil {
		return nil, taskErr // Already formatted as TaskRequestError
	}

	// --- Prepare LLM Call Parameters ---
	// Pass historyLength from params
	llmParams, err := prepareLLMParams(t, toolRegistry, sendParams.HistoryLength)
	if err != nil {
		// This function currently doesn't return an error, but could in the future
		// Update task to failed? For now, return generic error.
		updateTaskToFailed(store, t, "Failed to prepare LLM parameters", err)
		sendFinalStatusUpdate(updater, t.ID, t.Status, true)
		return nil, &task.TaskRequestError{Code: -32000, Message: "Failed to prepare LLM parameters", Data: err.Error()}
	}

	// --- Initiate Streaming Generation ---
	providerEventChan, err := llmProvider.Generate(llmParams)
	if err != nil {
		errnie.Error(err, "taskID", t.ID)
		updateTaskToFailed(store, t, "Failed to initiate LLM generation", err)
		sendFinalStatusUpdate(updater, t.ID, t.Status, true)
		return nil, &task.TaskRequestError{Code: -32003, Message: "Failed to initiate LLM generation", Data: err.Error()}
	}

	// Channel to receive the final result from the stream processing goroutine
	resultChan := make(chan streamProcessingResult, 1)

	// --- Launch Goroutine to Process Stream ---
	go processStreamEvents(sendParams.ID, providerEventChan, updater, resultChan)

	// --- Wait for Stream Processing and Handle Results ---
	errnie.Info("Waiting for stream processing result", "taskID", sendParams.ID)
	streamResult := <-resultChan
	errnie.Info("Received stream processing result", "taskID", sendParams.ID)

	// --- Handle Post-Stream Processing ---
	// This function now encapsulates handling stream errors, tool execution, final LLM call,
	// final task updates, and sending the final SSE.
	handlePostStreamProcessing(t, streamResult, llmProvider, llmParams, toolRegistry, store, updater)

	// --- Return Initial Confirmation ---
	// Return the status reflecting the *start* of processing.
	initialConfirmationStatus := task.TaskStatus{
		State:     task.TaskStateWorking,
		Timestamp: t.Status.Timestamp, // Use timestamp from initial update
		Message:   sendParams.Message,
	}
	return map[string]any{
		"status": initialConfirmationStatus,
	}, nil
}

// retrieveAndPrepareTask fetches the task and updates its status to Working.
func retrieveAndPrepareTask(store task.TaskStore, sendParams taskSendParams) (*task.Task, *task.TaskRequestError) {
	t, err := store.GetTask(sendParams.ID) // Use ID
	if err != nil {
		// Use A2A specific error code
		return nil, &task.TaskRequestError{Code: -32001, Message: "Task not found", Data: fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", sendParams.ID, err.Error())}
	}
	t.History = append(t.History, sendParams.Message)
	t.Status = task.TaskStatus{
		State:     task.TaskStateWorking,
		Timestamp: time.Now().Format(time.RFC3339),
		Message:   sendParams.Message,
	}

	// Store push notification config if provided in this send request
	if sendParams.PushNotification != nil {
		if t.Metadata == nil {
			t.Metadata = make(map[string]any)
		}
		t.Metadata["pushNotificationConfig"] = *sendParams.PushNotification // Dereference pointer
	}

	if err := store.UpdateTask(t); err != nil {
		errnie.Error("Failed to update task before streaming", "error", err, "taskID", t.ID)
		return nil, &task.TaskRequestError{Code: -32000, Message: "Failed to update task initially", Data: err.Error()}
	}
	return t, nil
}

// prepareLLMParams sets up the parameters for the initial streaming LLM call.
// Added historyLength argument.
func prepareLLMParams(t *task.Task, toolRegistry *tools.Registry, historyLength *int) (provider.ProviderParams, error) {
	// Determine history to use using the helper function
	historyForLLM := getHistorySlice(t.History, historyLength)

	// Convert potentially truncated history for the provider
	providerMessages := convertA2AMessagesToProviderMessages(historyForLLM)

	availableTools := make([]mcp.Tool, 0, len(toolRegistry.Tools))
	for _, registeredTool := range toolRegistry.Tools {
		availableTools = append(availableTools, registeredTool.Tool)
	}

	// Fetch these parameters from task metadata or a configuration service.
	// Using hardcoded values for now.
	llmParams := provider.ProviderParams{
		Model:       "gpt-4o", // Configurable: e.g., t.Config.Model or config.Get("default_model")
		Temperature: 0.7,      // Configurable
		TopP:        1.0,      // Configurable
		MaxTokens:   1024,     // Configurable
		Messages:    providerMessages,
		Tools:       availableTools,
		Stream:      true,
	}
	// Currently no error path, but return error for future flexibility
	return llmParams, nil
}

// Helper function to build content string from various message parts
func buildContentFromParts(parts []task.MessagePart) string {
	var contentBuilder strings.Builder
	for _, part := range parts {
		switch p := part.(type) {
		case task.TextPart:
			contentBuilder.WriteString(p.Text)
		case task.FilePart:
			// Simple representation, could be expanded (e.g., include file size, type)
			contentBuilder.WriteString(fmt.Sprintf("[File: %s]", p.File.Name))
		case task.DataPart:
			// Attempt to marshal data to JSON, fallback to string representation
			jsonData, err := json.Marshal(p.Data)
			if err != nil {
				contentBuilder.WriteString(fmt.Sprintf("[Data: %v]", p.Data)) // Fallback
			} else {
				contentBuilder.WriteString(fmt.Sprintf("[Data: %s]", string(jsonData)))
			}
		}
	}
	return contentBuilder.String()
}

// Convert task.Message history (A2A format) to provider.Message slice (LLM format)
func convertA2AMessagesToProviderMessages(a2aHistory []task.Message) []provider.Message {
	providerMessages := make([]provider.Message, 0, len(a2aHistory))
	for _, a2aMsg := range a2aHistory {
		// Build content using the helper function
		content := buildContentFromParts(a2aMsg.Parts)

		// Basic role mapping (expand if needed)
		var providerRole string
		switch a2aMsg.Role {
		case task.MessageRoleUser:
			providerRole = "user"
		case task.MessageRoleAgent:
			providerRole = "assistant"
		default:
			providerRole = string(a2aMsg.Role) // Pass through unknown roles
		}

		providerMessages = append(providerMessages, provider.Message{
			Role:    providerRole,
			Content: content,
			// ToolCalls and ToolResults are generally handled by the request/response flow
			// rather than being stored directly within the provider.Message content itself.
		})
	}
	return providerMessages
}

// processStreamEvents consumes events from the provider channel, sends artifact updates,
// and accumulates results, sending them back on the resultChan when done.
func processStreamEvents(
	taskID string,
	providerEventChan <-chan provider.ProviderEvent,
	updater StreamUpdateSender,
	resultChan chan<- streamProcessingResult,
) {
	errnie.Info("Starting stream processing goroutine", "taskID", taskID)
	defer errnie.Info("Exiting stream processing goroutine", "taskID", taskID)

	var accumulatedContent strings.Builder
	var accumulatedToolCalls []provider.PendingToolCall
	var assistantMessage provider.Message // Store the full assistant message if available
	var processingError error

	for event := range providerEventChan { // Read events until channel is closed
		errnie.Debug("Received provider event", "taskID", taskID)

		// --- Handle Potential Error Event First ---
		if event.Message.Role == "error" {
			errnie.Error("Received error event from provider", "taskID", taskID, "error", event.Message.Content)
			processingError = fmt.Errorf("provider stream error: %s", event.Message.Content)
			// Optionally break or continue depending on whether other events might follow an error
			// Assuming the stream terminates on error, we can break here.
			break
		}

		// Store the full assistant message if it's the container for tool calls
		// (Some providers might structure events differently)
		if event.Message.Role == "assistant" && len(event.ToolCalls) > 0 {
			assistantMessage = event.Message // Store the message requesting tools
		}

		// --- Handle Message Content ---
		if event.Message.Content != "" {
			accumulatedContent.WriteString(event.Message.Content)
			// Send artifact update via SSE
			artifactUpdate := task.TaskArtifactUpdateEvent{
				ID: taskID,
				Artifact: task.Artifact{
					Index:     0,     // Assuming one primary artifact stream
					Append:    true,  // Append chunks
					LastChunk: false, // Assume not last unless channel closes
					Parts:     []task.Part{{Text: event.Message.Content}},
				},
			}
			updater.SendTaskUpdate(taskID, artifactUpdate)
		}

		// --- Accumulate Tool Calls ---
		if len(event.ToolCalls) > 0 {
			errnie.Info("Received tool call request(s) in stream", "taskID", taskID, "count", len(event.ToolCalls))
			accumulatedToolCalls = append(accumulatedToolCalls, event.ToolCalls...)
		}
	}

	// --- Stream Finished ---
	errnie.Info("Provider event channel closed", "taskID", taskID)

	// Check for stream error *after* the loop (relevant for OpenAI provider)
	// This requires modification if the provider can send error events *during* the stream.
	// Example: if providerEventChan was type <-chan ResultOrError

	// If assistantMessage wasn't explicitly captured (e.g., only content chunks came),
	// and there were tool calls, synthesize an assistant message from accumulated content.
	if assistantMessage.Role == "" && len(accumulatedToolCalls) > 0 {
		assistantMessage = provider.Message{Role: "assistant", Content: accumulatedContent.String()}
	}

	// Send the accumulated result back
	resultChan <- streamProcessingResult{
		accumulatedContent:   accumulatedContent.String(),
		accumulatedToolCalls: accumulatedToolCalls,
		assistantMessage:     assistantMessage,
		err:                  processingError, // Send any error encountered during processing
	}
	close(resultChan) // Close channel after sending the single result
}

// handlePostStreamProcessing deals with the results after the LLM stream has finished.
func handlePostStreamProcessing(
	t *task.Task,
	streamResult streamProcessingResult,
	llmProvider provider.ProviderType,
	baseLLMParams provider.ProviderParams, // Pass the base params used for streaming
	toolRegistry *tools.Registry,
	store task.TaskStore,
	updater StreamUpdateSender,
) {
	// Handle stream processing error first
	if streamResult.err != nil {
		errnie.Error("Stream processing failed", "error", streamResult.err, "taskID", t.ID)
		updateTaskToFailed(store, t, "Stream processing error", streamResult.err)
		sendFinalStatusUpdate(updater, t.ID, t.Status, true)
		return // Task failed, nothing more to do
	}

	// Add the assistant's response from the stream to history
	addAssistantStreamResponseToHistory(t, streamResult)

	var finalAgentMessage task.Message
	var finalTaskState task.TaskState

	if len(streamResult.accumulatedToolCalls) > 0 {
		// Delegate tool execution and final response generation
		finalMsg, finalState, execErr := executeToolsAndGetResponse(
			t, streamResult.accumulatedToolCalls, toolRegistry, llmProvider, baseLLMParams, store,
		)
		finalAgentMessage = finalMsg // Use the message (success or error) from the execution function
		finalTaskState = finalState
		if execErr != nil {
			// Error was already logged, task updated by executeToolsAndGetResponse
			errnie.Warn("Tool execution or final response generation failed", "taskID", t.ID, "error", execErr)
		} else {
			finalTaskState = task.TaskStateCompleted // Explicitly set completed on success
		}
	} else {
		// No tool calls, the accumulated content is the final response
		finalAgentMessage = task.Message{
			Role: task.MessageRoleAgent,
			Parts: []task.MessagePart{
				task.TextPart{Type: "text", Text: streamResult.accumulatedContent},
			},
		}
		finalTaskState = task.TaskStateCompleted
	}

	// Update task with the determined final state and message
	updateFinalTaskStatus(store, t, finalAgentMessage, finalTaskState)

	// Send final SSE update
	sendFinalStatusUpdate(updater, t.ID, t.Status, true)
	errnie.Info("Post-stream processing complete", "taskID", t.ID, "finalState", t.Status.State)
}

// addAssistantStreamResponseToHistory adds the assistant message from the stream result to the task history.
func addAssistantStreamResponseToHistory(t *task.Task, streamResult streamProcessingResult) {
	// Only add if there was content OR tool calls were made (indicating an assistant turn)
	if streamResult.accumulatedContent != "" || len(streamResult.accumulatedToolCalls) > 0 {
		assistantMsgToStore := streamResult.assistantMessage
		// Synthesize if not explicitly provided but tools were called
		if assistantMsgToStore.Role == "" && len(streamResult.accumulatedToolCalls) > 0 {
			assistantMsgToStore = provider.Message{Role: "assistant", Content: streamResult.accumulatedContent}
		}
		// Convert provider.Message to task.Message before appending
		taskAssistantMsg := task.Message{
			Role: task.MessageRoleAgent,
			Parts: []task.MessagePart{
				task.TextPart{Type: "text", Text: assistantMsgToStore.Content},
			},
			// Tool call requests are implicitly handled by the flow (execution + subsequent tool messages)
			// and don't need explicit representation in the assistant's message parts in history.
		}
		// If there was text content OR tool calls were requested, add the message.
		if assistantMsgToStore.Content != "" || len(streamResult.accumulatedToolCalls) > 0 {
			t.History = append(t.History, taskAssistantMsg)
		}
	}
}

// executeToolsAndGetResponse handles the sequence of executing tools and getting a final LLM response.
func executeToolsAndGetResponse(
	t *task.Task,
	pendingCalls []provider.PendingToolCall,
	toolRegistry *tools.Registry,
	llmProvider provider.ProviderType,
	baseLLMParams provider.ProviderParams,
	store task.TaskStore,
) (finalMessage task.Message, finalState task.TaskState, err error) {
	errnie.Info("Executing accumulated tool calls", "taskID", t.ID, "count", len(pendingCalls))

	// Convert current task history back to provider format
	currentProviderMessages := convertA2AMessagesToProviderMessages(t.History)

	// Execute tools
	toolResultMessages, toolExecErr := executeTools(toolRegistry, pendingCalls) // Use existing helper
	if toolExecErr != nil {
		errnie.Error(toolExecErr, "taskID", t.ID)
		updateTaskToFailed(store, t, "Tool execution failed", toolExecErr)
		errMsg := task.Message{
			Role:  task.MessageRoleAgent,
			Parts: []task.MessagePart{task.TextPart{Type: "text", Text: fmt.Sprintf("Error during tool execution: %s", toolExecErr)}},
		}
		return errMsg, task.TaskStateFailed, toolExecErr // Return error message, failed state, and error
	}

	// Add tool results to provider messages
	currentProviderMessages = append(currentProviderMessages, toolResultMessages...)

	// Call LLM again (non-streaming)
	finalLLMParams := provider.ProviderParams{
		Model:            baseLLMParams.Model,
		Temperature:      baseLLMParams.Temperature,
		TopP:             baseLLMParams.TopP,
		MaxTokens:        baseLLMParams.MaxTokens,
		FrequencyPenalty: baseLLMParams.FrequencyPenalty, // Copy relevant fields
		PresencePenalty:  baseLLMParams.PresencePenalty,  // Copy relevant fields
		Messages:         currentProviderMessages,        // Use updated messages
		Tools:            nil,                            // No tools for final response
		Stream:           false,                          // Not streaming
		ResponseFormat:   baseLLMParams.ResponseFormat,   // Keep original format spec if any
	}

	finalEventChan, finalErr := llmProvider.Generate(finalLLMParams)
	if finalErr != nil {
		errnie.Error("Failed to get final LLM response after tools", "error", finalErr, "taskID", t.ID)
		updateTaskToFailed(store, t, "Failed final LLM call after tools", finalErr)
		errMsg := task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: "Error generating final response after tool use."}}}
		return errMsg, task.TaskStateFailed, finalErr
	}

	// Wait for the single final event
	var finalEvent provider.ProviderEvent
	select {
	case event, ok := <-finalEventChan:
		if !ok {
			errnie.Error("Final LLM response channel closed unexpectedly", "taskID", t.ID)
			llmErr := fmt.Errorf("LLM communication error: channel closed")
			updateTaskToFailed(store, t, llmErr.Error(), nil) // Pass error message directly
			errMsg := task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: "Error receiving final LLM response."}}}
			return errMsg, task.TaskStateFailed, llmErr
		}
		finalEvent = event
		// Convert final provider message to task message
		finalMsg := task.Message{
			Role: task.MessageRoleAgent,
			Parts: []task.MessagePart{
				task.TextPart{Type: "text", Text: finalEvent.Message.Content},
			},
		}
		return finalMsg, task.TaskStateCompleted, nil // Success

	case <-time.After(30 * time.Second): // Timeout for final response
		errnie.Error("Timeout waiting for final LLM response", "taskID", t.ID)
		timeoutErr := fmt.Errorf("LLM timeout waiting for final response")
		updateTaskToFailed(store, t, timeoutErr.Error(), nil) // Pass error message directly
		errMsg := task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: "Timeout receiving final LLM response."}}}
		return errMsg, task.TaskStateFailed, timeoutErr
	}
}

// executeSingleTool executes a single requested tool and returns the result message and any error.
func executeSingleTool(toolRegistry *tools.Registry, pendingCall provider.PendingToolCall) (provider.Message, error) {
	var resultMsg provider.Message
	var executionError error
	toolFound := false

	for _, registeredTool := range toolRegistry.Tools {
		if registeredTool.Tool.Name == pendingCall.Request.Params.Name {
			toolFound = true
			errnie.Info("Executing tool", "name", registeredTool.Tool.Name, "id", pendingCall.ID)

			// Pass a context derived from the overall request or task lifecycle
			// instead of context.Background() if cancellation/timeouts need propagation.
			toolResult, toolErr := registeredTool.Use(context.Background(), pendingCall.Request)

			resultMsg = provider.Message{
				ID:   pendingCall.ID, // Use the ID from the request for correlation
				Role: "tool",
				Name: registeredTool.Tool.Name,
			}

			if toolErr != nil {
				// Use the helper function to handle the error
				resultMsg, executionError = handleToolExecutionError(pendingCall, registeredTool.Tool.Name, toolErr)
			} else {
				// Extract text content using the helper function
				resultMsg.Content = processToolResultContent(toolResult)
			}
			break // Tool found and processed
		}
	}

	if !toolFound {
		// Use the helper function to handle the tool not found case
		return handleToolNotFound(pendingCall)
	}

	return resultMsg, executionError
}

// processToolResultContent extracts the text content from a tool execution result.
func processToolResultContent(toolResult *mcp.CallToolResult) string {
	if toolResult == nil || len(toolResult.Content) == 0 {
		return "[Tool produced no content]"
	}
	// Assuming the first part is the primary text content.
	// Adapt this logic if the structure of mcp.CallToolResult can vary more.
	if textContent, ok := toolResult.Content[0].(mcp.TextContent); ok {
		return textContent.Text
	}
	// Handle cases where the content isn't simple text or needs different processing.
	// For now, return a generic message.
	return "[Tool produced non-text content]"
}

// executeTools runs the necessary tools sequentially and returns the results as provider messages.
func executeTools(toolRegistry *tools.Registry, pendingCalls []provider.PendingToolCall) ([]provider.Message, error) {
	// Consider concurrent execution? For now, sequential.
	toolResultMessages := make([]provider.Message, 0, len(pendingCalls))
	var firstError error // Store the first error encountered

	for _, pendingCall := range pendingCalls {
		resultMsg, err := executeSingleTool(toolRegistry, pendingCall)
		if err != nil && firstError == nil {
			firstError = err
		}
		toolResultMessages = append(toolResultMessages, resultMsg)
	}

	return toolResultMessages, firstError
}

// updateFinalTaskStatus updates the task history and status in the store with the final result.
func updateFinalTaskStatus(store task.TaskStore, t *task.Task, finalMessage task.Message, finalState task.TaskState) {
	t.History = append(t.History, finalMessage) // Append final agent response
	t.Status = task.TaskStatus{
		State:     finalState,
		Timestamp: time.Now().Format(time.RFC3339),
		Message:   finalMessage, // Store final message in status
	}
	// Use helper to update the task, logging any error internally
	_ = updateTaskInStore(store, t, "task.send/updateFinalTaskStatus") // Ignoring the error here as the original code did.
	// The helper already logs the error.
}

// updateTaskToFailed updates the task status to failed in the store.
func updateTaskToFailed(store task.TaskStore, t *task.Task, message string, err error) {
	t.Status.State = task.TaskStateFailed
	t.Status.Timestamp = time.Now().Format(time.RFC3339)
	errorMsg := message
	if err != nil {
		errorMsg = fmt.Sprintf("%s: %s", message, err.Error())
	}
	// Append or replace the status message? Replacing for clarity of final error.
	t.Status.Message = task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: "ERROR: " + errorMsg}}}

	// Use helper to update the task, logging any error internally
	_ = updateTaskInStore(store, t, "task.send/updateTaskToFailed") // Ignoring the error here as the original code did.
	// The helper already logs the error.
}

// sendFinalStatusUpdate sends the final task status via SSE.
func sendFinalStatusUpdate(updater StreamUpdateSender, taskID string, status task.TaskStatus, final bool) {
	finalUpdate := task.TaskStatusUpdateEvent{
		ID:     taskID,
		Status: status,
		Final:  final,
	}
	// This function handles sending the update via the updater interface (which likely handles SSE).
	updater.SendTaskUpdate(taskID, finalUpdate)
}

// handleToolExecutionError formats the error message for a failed tool execution.
func handleToolExecutionError(pendingCall provider.PendingToolCall, toolName string, toolErr error) (provider.Message, error) {
	errnie.Error("Tool execution failed", "name", toolName, "id", pendingCall.ID, "error", toolErr)
	resultMsg := provider.Message{
		ID:      pendingCall.ID,
		Role:    "tool",
		Name:    toolName,
		Content: fmt.Sprintf("Error executing tool: %s", toolErr.Error()),
	}
	executionError := fmt.Errorf("tool '%s' failed: %w", toolName, toolErr)
	return resultMsg, executionError
}

// handleToolNotFound handles the case where a requested tool is not found in the registry.
func handleToolNotFound(pendingCall provider.PendingToolCall) (provider.Message, error) {
	toolName := pendingCall.Request.Params.Name
	errnie.Warn("LLM requested unknown tool", "name", toolName, "id", pendingCall.ID)
	resultMsg := provider.Message{
		ID:      pendingCall.ID,
		Role:    "tool",
		Name:    toolName,
		Content: fmt.Sprintf("Error: Tool '%s' not found.", toolName),
	}
	executionError := fmt.Errorf("tool '%s' not found", toolName)
	return resultMsg, executionError
}
