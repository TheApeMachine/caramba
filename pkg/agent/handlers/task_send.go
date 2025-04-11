package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service/types"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/utils"
)

// --- Example Structured Response Definition ---
// In a real app, move this to a shared types/models package
type ExampleStructuredResponse struct {
	Field1 string `json:"field_1" jsonschema_description:"Description for field 1"`
	Field2 int    `json:"field_2" jsonschema_description:"Description for field 2"`
}

// --- Schema Registry ---
// Maps a schema name requested by the client to a function that generates the schema
var responseSchemaRegistry = map[string]func() any{
	"example_response": func() any { return utils.GenerateSchema[ExampleStructuredResponse]() },
	// Add other response types here, e.g.:
	// "historical_computer": func() any { return utils.GenerateSchema[HistoricalComputer]() },
}

// taskSendParams defines the expected parameters for the task.send method
type taskSendParams struct {
	TaskID  string       `json:"task_id"` // Assuming A2A spec uses task_id
	Message task.Message `json:"message"` // Assuming message structure is defined in task package
	// Add other fields as needed by A2A spec (e.g., historyLength, metadata)
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

// Helper function to get metadata from a MessagePart
func getPartMetadata(part task.MessagePart) map[string]any {
	switch p := part.(type) {
	case task.TextPart:
		return p.Metadata
	case task.FilePart:
		return p.Metadata
	case task.DataPart:
		return p.Metadata
	default:
		return nil
	}
}

// Parses and validates the input parameters for task.send
func parseAndValidateTaskSendParams(params json.RawMessage) (taskSendParams, *task.TaskRequestError) {
	var sendParams taskSendParams
	if err := types.SimdUnmarshalJSON(params, &sendParams); err != nil {
		return taskSendParams{}, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.send",
			Data:    err.Error(),
		}
	}

	if sendParams.TaskID == "" {
		return taskSendParams{}, &task.TaskRequestError{
			Code:    -32602, // Invalid params
			Message: "Invalid params for task.send",
			Data:    "Missing required parameter: task_id",
		}
	}
	return sendParams, nil
}

// Retrieves a task from the store by ID
func getTaskFromStore(store task.TaskStore, taskID string) (*task.Task, *task.TaskRequestError) {
	t, err := store.GetTask(taskID)
	if err != nil {
		// Handle potential store errors (e.g., not found)
		return nil, &task.TaskRequestError{
			Code:    -32000, // Example: Application-specific error code for not found
			Message: "Task not found",
			Data:    fmt.Sprintf("Task with ID '%s' not found or error retrieving: %s", taskID, err.Error()),
		}
	}
	return t, nil
}

// Prepares the ProviderParams for the LLM call
func prepareProviderParams(a2aHistory []task.Message, toolRegistry *tools.Registry, userMessage task.Message) provider.ProviderParams {
	// Convert history for the provider
	providerMessages := convertA2AMessagesToProviderMessages(a2aHistory)

	// Get available tools from the registry
	availableTools := make([]mcp.Tool, 0, len(toolRegistry.Tools))
	for _, registeredTool := range toolRegistry.Tools {
		availableTools = append(availableTools, registeredTool.Tool)
	}

	// Base LLM parameters
	llmParams := provider.ProviderParams{
		Model:            tweaker.GetModel(tweaker.GetProvider()),
		Temperature:      tweaker.GetTemperature(),
		TopP:             tweaker.GetTopP(),
		MaxTokens:        tweaker.GetMaxTokens(),
		FrequencyPenalty: tweaker.GetFrequencyPenalty(),
		PresencePenalty:  tweaker.GetPresencePenalty(),
		Stream:           true, // Assuming streaming is always desired here
		Messages:         providerMessages,
		Tools:            availableTools,
	}

	// Determine the response format based on user message metadata
	llmParams.ResponseFormat = determineResponseFormat(userMessage)

	return llmParams
}

// determines the desired response format (if any) based on metadata in the user message parts.
func determineResponseFormat(userMessage task.Message) provider.ResponseFormat {
	var foundFormat *provider.ResponseFormat = nil

	for _, part := range userMessage.Parts {
		metadata := getPartMetadata(part)
		if metadata == nil {
			continue
		}

		// PRIORITY 1: Check for Named Schema
		namedFormat, found := findNamedResponseFormat(metadata)
		if found {
			foundFormat = namedFormat
			break // Found the highest priority format, stop checking parts
		}
	}

	// PRIORITY 2: Check for Inline Schema (only if named wasn't found)
	if foundFormat == nil {
		for _, part := range userMessage.Parts {
			metadata := getPartMetadata(part)
			if metadata == nil {
				continue
			}

			inlineFormat, found := findInlineResponseFormat(metadata)
			if found {
				foundFormat = inlineFormat
				break // Found an inline format, stop checking parts
			}
		}
	}

	if foundFormat != nil {
		return *foundFormat
	}

	// Return a zero-value ResponseFormat if none was determined
	return provider.ResponseFormat{}
}

// findNamedResponseFormat checks metadata for a named response schema and returns it if found.
func findNamedResponseFormat(metadata map[string]any) (*provider.ResponseFormat, bool) {
	schemaName, ok := metadata["responseSchemaName"].(string)
	if !ok {
		return nil, false // No name specified
	}

	schemaGenerator, found := responseSchemaRegistry[schemaName]
	if !found {
		// Optional: Log warning if name provided but not found
		fmt.Printf("Warning: Requested response schema name '%s' not found in registry\n", schemaName)
		return nil, false // Name specified but not found in registry
	}

	generatedSchema := schemaGenerator()
	return &provider.ResponseFormat{
		Name:        schemaName,
		Description: fmt.Sprintf("Structured response for %s", schemaName),
		Schema:      generatedSchema,
		Strict:      true,
	}, true // Found and created successfully
}

// findInlineResponseFormat checks metadata for an inline JSON schema and returns it if found.
func findInlineResponseFormat(metadata map[string]any) (*provider.ResponseFormat, bool) {
	mimeType, mimeOk := metadata["mimeType"].(string)
	schema, schemaOk := metadata["schema"]

	if mimeOk && mimeType == "application/json" && schemaOk {
		// Found an inline JSON schema request
		return &provider.ResponseFormat{
			// Name/Description could potentially be extracted from metadata too if desired
			Schema: schema,
			Strict: true,
		}, true
	}
	return nil, false // No inline schema found
}

// Calls the LLM provider and waits for the event, handling channel communication and timeout.
func callLLMAndReceiveEvent(llmProvider provider.ProviderType, params provider.ProviderParams, taskID string) (provider.ProviderEvent, *task.TaskRequestError) {
	// Call the LLM provider
	providerEventChan, err := llmProvider.Generate(params)
	if err != nil {
		errnie.Error("LLM generation failed", "taskID", taskID, "error", err)
		return provider.ProviderEvent{}, &task.TaskRequestError{Code: -32003, Message: "LLM generation failed", Data: err.Error()}
	}

	// Receive the actual event from the channel with timeout
	var providerEvent provider.ProviderEvent
	select {
	case event, ok := <-providerEventChan:
		if !ok {
			errnie.Error("LLM provider channel closed unexpectedly", "taskID", taskID)
			return provider.ProviderEvent{}, &task.TaskRequestError{Code: -32004, Message: "LLM provider communication error", Data: "Channel closed"}
		}
		providerEvent = event
	case <-time.After(30 * time.Second): // Add a timeout
		errnie.Error("Timeout waiting for LLM provider event", "taskID", taskID)
		return provider.ProviderEvent{}, &task.TaskRequestError{Code: -32005, Message: "LLM provider timeout", Data: "Timeout waiting for response"}
	}
	return providerEvent, nil
}

// executeSingleToolConcurrently handles the execution of a single tool within a goroutine.
func executeSingleToolConcurrently(ctx context.Context, toolRegistry *tools.Registry, call provider.PendingToolCall, taskID string, wg *sync.WaitGroup, resultChan chan<- provider.Message) {
	defer wg.Done()
	var resultMsg provider.Message
	toolFound := false

	for _, registeredTool := range toolRegistry.Tools {
		if registeredTool.Tool.Name == call.Request.Params.Name {
			toolFound = true
			errnie.Info("Executing tool", "name", registeredTool.Tool.Name, "id", call.ID, "taskID", taskID)
			toolResult, toolErr := registeredTool.Use(ctx, call.Request) // Pass down the main context

			resultMsg = provider.Message{
				ID:   call.ID, // Use the ID from the request for correlation
				Role: "tool",
				Name: registeredTool.Tool.Name,
			}
			if toolErr != nil {
				// Use the helper function to format the error message
				resultMsg = handleToolExecutionErrorConcurrent(call, registeredTool.Tool.Name, taskID, toolErr)
			} else {
				// Use the helper function to process the result content
				resultMsg.Content = processToolResultContentConcurrent(toolResult, registeredTool.Tool.Name, call.ID, taskID)
			}
			break // Found and processed the tool, exit inner loop
		}
	}

	if !toolFound {
		// Use the helper function to handle the tool not found case
		resultMsg = handleToolNotFoundConcurrent(call, taskID)
	}
	resultChan <- resultMsg // Send result to channel
}

// processToolResultContentConcurrent extracts the text content from a tool execution result, logging warnings for unexpected content.
func processToolResultContentConcurrent(toolResult *mcp.CallToolResult, toolName, callID, taskID string) string {
	if toolResult == nil || len(toolResult.Content) == 0 {
		errnie.Warn("Tool produced no result content", "name", toolName, "id", callID, "taskID", taskID)
		return "[Tool produced no result content]"
	}

	if textContent, ok := toolResult.Content[0].(mcp.TextContent); ok {
		return textContent.Text
	}

	errnie.Warn("Tool result content was not text", "name", toolName, "id", callID, "taskID", taskID)
	return "[Tool produced non-text content]"
}

// handleToolExecutionErrorConcurrent formats the error message for a failed concurrent tool execution.
func handleToolExecutionErrorConcurrent(call provider.PendingToolCall, toolName string, taskID string, toolErr error) provider.Message {
	errnie.Error("Tool execution failed", "name", toolName, "id", call.ID, "taskID", taskID, "error", toolErr)
	return provider.Message{
		ID:      call.ID,
		Role:    "tool",
		Name:    toolName,
		Content: fmt.Sprintf("Error executing tool: %s", toolErr.Error()),
	}
}

// handleToolNotFoundConcurrent handles the case where a requested tool is not found during concurrent execution.
func handleToolNotFoundConcurrent(call provider.PendingToolCall, taskID string) provider.Message {
	toolName := call.Request.Params.Name
	errnie.Info("LLM requested unknown tool", "name", toolName, "id", call.ID, "taskID", taskID)
	return provider.Message{
		ID:      call.ID,
		Role:    "tool",
		Name:    toolName,
		Content: fmt.Sprintf("Error: Tool '%s' not found.", toolName),
	}
}

// Executes requested tools concurrently and collects their results.
func executeToolsAndCollectResults(ctx context.Context, toolRegistry *tools.Registry, toolCalls []provider.PendingToolCall, taskID string) []provider.Message {
	toolResultMessages := []provider.Message{}
	var wg sync.WaitGroup
	resultChan := make(chan provider.Message, len(toolCalls))

	for _, pendingCall := range toolCalls {
		wg.Add(1)
		// Pass pendingCall by value to the goroutine
		go executeSingleToolConcurrently(ctx, toolRegistry, pendingCall, taskID, &wg, resultChan)
	}

	// Close the channel once all goroutines are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results from the channel
	for res := range resultChan {
		toolResultMessages = append(toolResultMessages, res)
	}

	return toolResultMessages
}

// Executes the LLM call, handling the tool execution loop if necessary.
func executeLLMWithTools(ctx context.Context, llmProvider provider.ProviderType, toolRegistry *tools.Registry, initialParams provider.ProviderParams, taskID string) (provider.ProviderEvent, []provider.Message, *task.TaskRequestError) {
	var finalProviderEvent provider.ProviderEvent
	currentMessages := initialParams.Messages
	llmParams := initialParams // Start with initial params

	for i := 0; i < 5; i++ { // Limit iterations to prevent infinite loops
		// Prepare params for this iteration
		currentLLMParams := llmParams               // Copy base params from the initial/previous iteration
		currentLLMParams.Messages = currentMessages // Update messages
		// *** Crucially, only send tools on the FIRST request in the loop ***
		if i > 0 {
			currentLLMParams.Tools = nil
		}

		// Call the LLM provider using the helper function
		providerEvent, reqErr := callLLMAndReceiveEvent(llmProvider, currentLLMParams, taskID)
		if reqErr != nil {
			return provider.ProviderEvent{}, currentMessages, reqErr
		}

		// If no tool calls, this is the final event
		if len(providerEvent.ToolCalls) == 0 {
			finalProviderEvent = providerEvent
			break
		}

		// --- Tool Execution Step --- (Remains complex, but now isolated)
		// Add the assistant message requesting tool calls to the history
		assistantMessage := providerEvent.Message // This message contains the tool call *requests*
		currentMessages = append(currentMessages, assistantMessage)

		// Execute tools and collect results using the helper function
		toolResultMessages := executeToolsAndCollectResults(ctx, toolRegistry, providerEvent.ToolCalls, taskID)

		// Add tool results to messages for the next LLM iteration
		currentMessages = append(currentMessages, toolResultMessages...)

		// Safety break if loop runs too long
		if i == 4 {
			errnie.Error("Tool execution loop reached max iterations", "taskID", taskID)
			// Use the last assistant message *before* attempting to process results as the final event
			finalProviderEvent = providerEvent
			break
			// Alternatively, create an error response:
			// finalProviderEvent = provider.ProviderEvent{Message: provider.Message{Role: "assistant", Content: "Error: Tool execution took too many steps."}}
			// break
		}
	}

	// Return the final event and the final message history
	return finalProviderEvent, currentMessages, nil
}

// Updates the task with the final agent response and saves it to the store.
func updateTaskWithFinalResponse(store task.TaskStore, t *task.Task, finalProviderEvent provider.ProviderEvent) *task.TaskRequestError {
	// Create agent response message from the final event
	agentResponse := task.Message{
		Role: task.MessageRoleAgent,
		Parts: []task.MessagePart{
			task.TextPart{
				Type: "text",
				Text: finalProviderEvent.Message.Content,
			},
		},
		// Final providerEvent.ToolCalls are not mapped back
	}

	// Add final agent response to history
	t.History = append(t.History, agentResponse)

	// Determine final task state
	finalState := task.TaskStateCompleted
	if finalProviderEvent.Message.Content == "" && len(finalProviderEvent.ToolCalls) == 0 {
		// Handle the error case from max iterations if needed (e.g., set to Failed)
		errnie.Info("Tool loop ended with potentially empty final event", "taskID", t.ID)
		// finalState = task.TaskStateFailed // Example: Mark as failed if loop maxed out
	}

	t.Status = task.TaskStatus{
		State:     finalState,
		Timestamp: time.Now().Format(time.RFC3339),
		Message:   agentResponse, // Store the final agent message
	}

	// Update the task in the store
	if err := store.UpdateTask(t); err != nil {
		return &task.TaskRequestError{
			Code:    -32000, // Store update error
			Message: "Failed to update task after LLM response",
			Data:    err.Error(),
		}
	}

	return nil // Success
}

// Modified signature to accept LLM provider AND tool registry
func HandleTaskSend(ctx context.Context, store task.TaskStore, llmProvider provider.ProviderType, toolRegistry *tools.Registry, params json.RawMessage) (interface{}, *task.TaskRequestError) {
	// Parse and validate parameters
	sendParams, reqErr := parseAndValidateTaskSendParams(params)
	if reqErr != nil {
		return nil, reqErr
	}

	// Retrieve task from store
	t, reqErr := getTaskFromStore(store, sendParams.TaskID)
	if reqErr != nil {
		return nil, reqErr
	}

	// Add user message to task history
	t.History = append(t.History, sendParams.Message)

	// Prepare LLM parameters using the helper function
	llmParams := prepareProviderParams(t.History, toolRegistry, sendParams.Message)

	// Execute LLM call, handling tool loop if necessary
	finalProviderEvent, _, reqErr := executeLLMWithTools(ctx, llmProvider, toolRegistry, llmParams, t.ID)
	if reqErr != nil {
		// Update task status to Failed based on the error from the loop
		t.Status = task.TaskStatus{
			State:     task.TaskStateFailed,
			Timestamp: time.Now().Format(time.RFC3339),
			Message:   task.Message{Role: task.MessageRoleAgent, Parts: []task.MessagePart{task.TextPart{Type: "text", Text: reqErr.Message + ": " + fmt.Sprintf("%v", reqErr.Data)}}}, // Include error details
		}
		_ = store.UpdateTask(t) // Attempt to update task even on failure
		return nil, reqErr
	}

	// Update task with the final response
	if reqErr := updateTaskWithFinalResponse(store, t, finalProviderEvent); reqErr != nil {
		return nil, reqErr
	}

	// SSE updates are typically handled by a separate mechanism or service
	// tied to task status changes, not directly within this handler.

	// Return confirmation (updated status)
	return map[string]interface{}{
		"status": t.Status,
	}, nil
}
