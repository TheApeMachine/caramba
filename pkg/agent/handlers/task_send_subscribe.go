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

type SendSubscribeHandler struct {
	ctx          fiber.Ctx
	store        task.TaskStore
	llmProvider  provider.ProviderType
	toolRegistry *tools.Registry
	err          error
}

func NewSendSubscriberHandler(
	ctx fiber.Ctx,
	store task.TaskStore,
	llmProvider provider.ProviderType,
	toolRegistry *tools.Registry,
) *SendSubscribeHandler {
	return &SendSubscribeHandler{
		ctx:          ctx,
		store:        store,
		llmProvider:  llmProvider,
		toolRegistry: toolRegistry,
	}
}

// HandleTaskSendSubscribe handles the tasks/sendSubscribe method, streaming SSE results directly.
func (handler *SendSubscribeHandler) HandleRequest(
	params json.RawMessage, requestID any,
) error {
	// Step 1: Parse parameters and retrieve task
	t, _, err := handler.parseParamsAndGetTask(params)
	if err != nil {
		return err
	}

	// Step 2: Setup LLM parameters and provider
	llmParams, providerEventChan, err := handler.setupLLMAndProvider(t)
	if err != nil {
		return err
	}

	// Step 3: Configure SSE headers and start streaming
	return handler.startSSEStream(t, llmParams, providerEventChan, requestID)
}

// parseParamsAndGetTask handles parameter validation and task retrieval
func (handler *SendSubscribeHandler) parseParamsAndGetTask(
	params json.RawMessage,
) (*task.Task, string, error) {
	var sendParams taskSendParams
	methodName := "tasks/sendSubscribe"

	if taskErr := parseAndValidateParams(params, &sendParams, methodName); taskErr != nil {
		return nil, methodName, taskErr
	}

	if sendParams.ID == "" {
		return nil, methodName, task.NewMissingParamError(methodName, "id")
	}

	t, taskErr := retrieveAndPrepareTask(handler.store, sendParams)
	if taskErr != nil {
		return nil, methodName, taskErr
	}

	return t, methodName, nil
}

// setupLLMAndProvider prepares LLM parameters and initializes the provider
func (handler *SendSubscribeHandler) setupLLMAndProvider(
	t *task.Task,
) (provider.ProviderParams, <-chan provider.ProviderEvent, error) {
	var llmParams provider.ProviderParams
	var providerEventChan <-chan provider.ProviderEvent

	// Prepare LLM parameters
	if llmParams, handler.err = prepareLLMParams(
		t, handler.toolRegistry, nil, // Use default history length
	); handler.err != nil {
		updateTaskToFailed(
			handler.store, t, "Failed to prepare LLM parameters", handler.err,
		)
		return llmParams, nil, &task.TaskRequestError{
			Code:    -32000,
			Message: "Failed to prepare LLM parameters",
			Data:    handler.err.Error(),
		}
	}

	// Initialize provider
	if providerEventChan, handler.err = handler.llmProvider.Generate(llmParams); handler.err != nil {
		errnie.Error(handler.err, "taskID", t.ID)
		updateTaskToFailed(
			handler.store, t, "Failed to initiate LLM generation", handler.err,
		)
		return llmParams, nil, &task.TaskRequestError{
			Code:    -32003,
			Message: "Failed to initiate LLM generation",
			Data:    handler.err.Error(),
		}
	}

	return llmParams, providerEventChan, nil
}

// startSSEStream sets up SSE headers and starts the streaming process
func (handler *SendSubscribeHandler) startSSEStream(
	t *task.Task,
	llmParams provider.ProviderParams,
	providerEventChan <-chan provider.ProviderEvent,
	requestID any,
) error {
	// Set SSE Headers
	handler.ctx.Set("Content-Type", "text/event-stream")
	handler.ctx.Set("Cache-Control", "no-cache")
	handler.ctx.Set("Connection", "keep-alive")

	// Start streaming
	streamErr := handler.ctx.SendStreamWriter(func(w *bufio.Writer) {
		handler.handleStreamEvents(w, t, llmParams, providerEventChan, requestID)
	})

	// Handle SendStreamWriter errors
	if streamErr != nil {
		errnie.Error("SendStreamWriter failed", "error", streamErr, "taskID", t.ID)
		handler.ensureTaskFailedStatus(t, streamErr)
	}

	return nil
}

// handleStreamEvents processes events from the provider and sends SSE updates
func (handler *SendSubscribeHandler) handleStreamEvents(
	w *bufio.Writer,
	t *task.Task,
	llmParams provider.ProviderParams,
	providerEventChan <-chan provider.ProviderEvent,
	requestID any,
) {
	errnie.Info("Starting SSE stream writer", "taskID", t.ID, "requestID", requestID)
	defer errnie.Info("Exiting SSE stream writer", "taskID", t.ID)

	// Initialize stream state
	state := handler.initStreamState(w, t, requestID)
	if state.streamErr != nil {
		return
	}

	// Process provider events
	handler.processProviderEvents(state, t, providerEventChan)

	// Handle end of stream
	handler.handleEndOfStream(state, t, llmParams, w, requestID)
}

// streamState contains the accumulated state during streaming
type streamState struct {
	accumulatedContent   strings.Builder
	accumulatedToolCalls []provider.PendingToolCall
	assistantMessage     provider.Message
	streamErr            error
}

// initStreamState sets up initial stream state and sends initial status
func (handler *SendSubscribeHandler) initStreamState(
	w *bufio.Writer,
	t *task.Task,
	requestID any,
) *streamState {
	state := &streamState{}

	// Send initial status update
	initialStatusEvent := task.TaskStatusUpdateEvent{
		ID: t.ID,
		Status: task.TaskStatus{
			State:     task.TaskStateWorking,
			Timestamp: t.Status.Timestamp,
		},
		Final: false,
	}

	if err := handler.writeSSEEvent(
		w, requestID, initialStatusEvent,
	); err != nil {
		state.streamErr = errnie.New(errnie.WithError(err))
		updateTaskToFailed(handler.store, t, "Failed to write initial SSE status", err)
	}

	return state
}

// processProviderEvents handles incoming events from the provider
func (handler *SendSubscribeHandler) processProviderEvents(
	state *streamState,
	t *task.Task,
	providerEventChan <-chan provider.ProviderEvent,
) {
	isStreaming := true

	for isStreaming && state.streamErr == nil {
		for event := range providerEventChan {
			errnie.Debug("Received provider event", "taskID", t.ID)

			if state.streamErr = handler.handleProviderEvent(state, t, event); state.streamErr != nil {
				isStreaming = false
				break
			}
		}

		isStreaming = false

		// Synthesize assistant message if needed
		if state.assistantMessage.Role == "" && len(state.accumulatedToolCalls) > 0 {
			state.assistantMessage = provider.Message{
				Role:    "assistant",
				Content: state.accumulatedContent.String(),
			}
		}

		// Add to history before tool execution
		addAssistantStreamResponseToHistory(t, streamProcessingResult{
			accumulatedContent:   state.accumulatedContent.String(),
			accumulatedToolCalls: state.accumulatedToolCalls,
			assistantMessage:     state.assistantMessage,
			err:                  nil,
		})
	}
}

// handleProviderEvent processes a single event from the provider
func (handler *SendSubscribeHandler) handleProviderEvent(
	state *streamState,
	t *task.Task,
	event provider.ProviderEvent,
) error {
	// Check for error events
	if event.Message.Role == "error" {
		errnie.Error("Received error event from provider", "taskID", t.ID, "error", event.Message.Content)
		return fmt.Errorf("provider stream error: %s", event.Message.Content)
	}

	// Store assistant message with tool calls
	if event.Message.Role == "assistant" && len(event.ToolCalls) > 0 {
		state.assistantMessage = event.Message
	}

	// Handle message content
	if event.Message.Content != "" {
		state.accumulatedContent.WriteString(event.Message.Content)
	}

	// Accumulate tool calls
	if len(event.ToolCalls) > 0 {
		errnie.Info("Received tool call request(s)", "taskID", t.ID, "count", len(event.ToolCalls))
		state.accumulatedToolCalls = append(state.accumulatedToolCalls, event.ToolCalls...)
	}

	return nil
}

// handleEndOfStream processes the final state after streaming completes
func (handler *SendSubscribeHandler) handleEndOfStream(
	state *streamState,
	t *task.Task,
	llmParams provider.ProviderParams,
	w *bufio.Writer,
	requestID any,
) {
	finalAgentMessage := task.Message{}
	finalTaskState := task.TaskStateFailed // Default to failed

	if state.streamErr != nil {
		// Handle stream error
		finalAgentMessage, finalTaskState = handler.handleStreamError(t, state.streamErr)
	} else if len(state.accumulatedToolCalls) > 0 {
		// Execute tools
		finalAgentMessage, finalTaskState = handler.executeToolcall(
			finalAgentMessage, t, state.accumulatedToolCalls, llmParams, w, requestID,
		)
	} else {
		// Stream finished successfully
		finalTaskState, finalAgentMessage = handler.streamFinished(
			finalAgentMessage, state.accumulatedContent, t, w, requestID,
		)
	}

	// Update task status and send final SSE event
	handler.finalizeTasks(t, finalAgentMessage, finalTaskState, w, requestID)
}

// handleStreamError processes errors from the stream
func (handler *SendSubscribeHandler) handleStreamError(
	t *task.Task,
	streamErr error,
) (task.Message, task.TaskState) {
	errnie.Error("Error processing provider stream", "error", streamErr, "taskID", t.ID)
	updateTaskToFailed(handler.store, t, "Provider stream error", streamErr)
	return task.Message{
		Role:  task.MessageRoleAgent,
		Parts: []task.MessagePart{task.TextPart{Text: "ERROR: " + streamErr.Error()}},
	}, task.TaskStateFailed
}

// finalizeTasks updates the task status and sends the final SSE event
func (handler *SendSubscribeHandler) finalizeTasks(
	t *task.Task,
	finalAgentMessage task.Message,
	finalTaskState task.TaskState,
	w *bufio.Writer,
	requestID any,
) {
	updateFinalTaskStatus(handler.store, t, finalAgentMessage, finalTaskState)

	finalStatusEvent := task.TaskStatusUpdateEvent{
		ID:     t.ID,
		Status: t.Status,
		Final:  true,
	}

	if err := handler.writeSSEEvent(
		w, requestID, finalStatusEvent,
	); err != nil {
		errnie.Error("Failed to write final SSE status", "error", err, "taskID", t.ID)
	}
}

// ensureTaskFailedStatus ensures task is marked as failed after stream errors
func (handler *SendSubscribeHandler) ensureTaskFailedStatus(t *task.Task, streamErr error) {
	currentTask, _ := handler.store.GetTask(t.ID)
	if currentTask != nil &&
		currentTask.Status.State != task.TaskStateFailed &&
		currentTask.Status.State != task.TaskStateCanceled {
		updateTaskToFailed(handler.store, t, "SSE stream writer failed", streamErr)
	}
}

func (handler *SendSubscribeHandler) streamFinished(
	finalAgentMessage task.Message,
	accumulatedContent strings.Builder,
	t *task.Task,
	w *bufio.Writer,
	requestID any,
) (task.TaskState, task.Message) {
	finalTaskState := task.TaskStateCompleted
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
	if err := handler.writeSSEEvent(
		w, requestID, finalArtifactUpdate,
	); err != nil {
		errnie.Error("Failed to write final SSE artifact", "error", err, "taskID", t.ID)
		updateTaskToFailed(handler.store, t, "Failed to write final artifact SSE", err)
		finalTaskState = task.TaskStateFailed // Mark failed if write fails
	}
	return finalTaskState, finalAgentMessage
}

func (handler *SendSubscribeHandler) executeToolcall(
	finalAgentMessage task.Message,
	t *task.Task,
	accumulatedToolCalls []provider.PendingToolCall,
	llmParams provider.ProviderParams,
	w *bufio.Writer,
	requestID any,
) (task.Message, task.TaskState) {
	var toolExecErr error
	var finalTaskState task.TaskState

	finalAgentMessage, finalTaskState, toolExecErr = executeToolsAndGetResponse(
		t, accumulatedToolCalls, handler.toolRegistry, handler.llmProvider, llmParams, handler.store,
	)
	if toolExecErr != nil {
		errnie.Warn("Tool execution or final response failed in sendSubscribe", "taskID", t.ID, "error", toolExecErr)
	} else {
		finalTaskState = task.TaskStateCompleted

		finalParts := make([]task.Part, len(finalAgentMessage.Parts))
		for i, p := range finalAgentMessage.Parts {
			finalParts[i] = task.Part{Text: fmt.Sprintf("[Unsupported Part: %T]", p)} // Assign fallback struct value
		}

		finalArtifactUpdate := task.TaskArtifactUpdateEvent{
			ID: t.ID,
			Artifact: task.Artifact{
				Index:     0,
				Append:    false,
				LastChunk: true,
				Parts:     finalParts,
			},
		}
		if err := handler.writeSSEEvent(
			w,
			requestID,
			finalArtifactUpdate,
		); err != nil {
			errnie.Error("Failed to write final SSE artifact after tools", "error", err, "taskID", t.ID)

			updateTaskToFailed(
				handler.store, t, "Failed to write final artifact SSE", err,
			)

			finalTaskState = task.TaskStateFailed
		}
	}
	return finalAgentMessage, finalTaskState
}

// writeSSEEvent formats and writes a JSON-RPC structured SSE event.
func (handler *SendSubscribeHandler) writeSSEEvent(w *bufio.Writer, requestID any, payload any) error {
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
