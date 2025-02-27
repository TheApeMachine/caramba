package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/briandowns/spinner"
	"github.com/theapemachine/caramba/pkg/output"
)

// IterationManager handles agent iteration logic.
type IterationManager struct {
	agent             Agent
	toolManager       *ToolManager
	responseProcessor *ResponseProcessor
	workflowManager   *WorkflowManager
}

// NewIterationManager creates a new IterationManager.
func NewIterationManager(
	agent Agent,
) *IterationManager {
	return &IterationManager{
		agent:             agent,
		toolManager:       NewToolManager(),
		responseProcessor: NewResponseProcessor(),
		workflowManager:   NewWorkflowManager(),
	}
}

func (im *IterationManager) SetWorkflow(workflow *Workflow) {
	im.workflowManager.SetWorkflow(workflow)
}

// Run handles the iteration loop, either in streaming or non-streaming mode.
func (im *IterationManager) Run(
	ctx context.Context,
	initialMsg LLMMessage,
	params *LLMParams,
	streaming bool,
) error {
	var response strings.Builder

	iteration := 0
	iterMsg := LLMMessage{
		Role:    initialMsg.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, initialMsg.Content),
	}

	for iteration < im.agent.IterationLimit() {
		// Add the current message to the conversation.
		params.Messages = append(params.Messages, iterMsg)

		// Sync tools with LLM params.
		params.Tools = im.toolManager.GetTools()

		// Show the spinner if needed.
		iterationStr := fmt.Sprintf("Iteration %d/%d", iteration+1, im.agent.IterationLimit())
		thinkingSpinner := output.StartSpinner(fmt.Sprintf("%s: Agent thinking", iterationStr))

		var iterationResponse string
		var err error

		if streaming {
			iterationResponse, err = im.handleStreamingIteration(
				ctx, thinkingSpinner, *params, im.agent.LLM(),
			)
		} else {
			iterationResponse, err = im.handleNonStreamingIteration(
				ctx, thinkingSpinner, *params, im.agent.LLM(),
			)
		}

		if err != nil {
			return err
		}

		// Append response
		response.WriteString(iterationResponse)

		// Next iteration
		iteration++
		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, response.String()),
		}
	}

	return nil
}

// handleNonStreamingIteration processes a single non-streaming iteration.
func (im *IterationManager) handleNonStreamingIteration(
	ctx context.Context,
	thinkingSpinner *spinner.Spinner,
	params LLMParams,
	llm LLMProvider,
) (string, error) {
	res := llm.GenerateResponse(ctx, params)
	if res.Error != nil {
		output.StopSpinner(thinkingSpinner, "")
		output.Error("LLM response generation failed", res.Error)
		return "", res.Error
	}

	var responseWithToolResults strings.Builder
	responseWithToolResults.WriteString(res.Content)

	// Handle tool calls if any
	if len(res.ToolCalls) > 0 {
		output.StopSpinner(thinkingSpinner, fmt.Sprintf("Agent is using %d tools", len(res.ToolCalls)))
		for _, toolCall := range res.ToolCalls {
			output.Action("agent", "tool_call", toolCall.Name)
		}

		// Execute tools and add results to the response
		toolResults := im.toolManager.ExecuteToolCalls(ctx, res.ToolCalls)
		responseWithToolResults.WriteString(toolResults)
	} else {
		output.StopSpinner(thinkingSpinner, "Agent completed thinking")
	}

	return responseWithToolResults.String(), nil
}

// handleStreamingIteration processes a single streaming iteration.
func (im *IterationManager) handleStreamingIteration(
	ctx context.Context,
	thinkingSpinner *spinner.Spinner,
	params LLMParams,
	llm LLMProvider,
) (string, error) {
	output.StopSpinner(thinkingSpinner, "Agent is responding in real-time:")
	output.Info("Streaming response begins:")
	fmt.Println(strings.Repeat("-", 40))

	// Accumulate the chunks
	var streamedResponse strings.Builder
	toolCallsFound := false
	var toolCalls []ToolCall

	// Process the stream
	streamChan := llm.StreamResponse(ctx, params)
	for chunk := range streamChan {
		if chunk.Error != nil {
			output.Error("Streaming failed", chunk.Error)
			return "", chunk.Error
		}

		// Handle tool calls in the chunk
		if len(chunk.ToolCalls) > 0 {
			toolCallsFound = true
			toolCalls = chunk.ToolCalls
			im.toolManager.LogToolCalls(toolCalls)
		} else if chunk.Content != "" {
			// Process and display content
			content := im.responseProcessor.ProcessChunkContent(chunk.Content)
			if content != "" {
				formatted := im.responseProcessor.FormatStreamedContent(content)
				fmt.Print(formatted)
				streamedResponse.WriteString(content)
			}
		}
	}

	fmt.Println()
	fmt.Println(strings.Repeat("-", 40))
	output.Info("Streaming response complete")

	// Execute tool calls if found
	if toolCallsFound {
		toolResults := im.toolManager.ExecuteToolCalls(ctx, toolCalls)
		streamedResponse.WriteString(toolResults)
	}

	return streamedResponse.String(), nil
}
