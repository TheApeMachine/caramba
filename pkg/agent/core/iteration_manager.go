package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// IterationManager handles agent iteration logic.
type IterationManager struct {
	logger            *output.Logger
	hub               *hub.Queue
	agent             Agent
	responseProcessor *ResponseProcessor
	workflowManager   *WorkflowManager
}

// NewIterationManager creates a new IterationManager.
func NewIterationManager(
	agent Agent,
) *IterationManager {
	return &IterationManager{
		logger:            output.NewLogger(),
		hub:               hub.NewQueue(),
		agent:             agent,
		responseProcessor: NewResponseProcessor(),
		workflowManager:   NewWorkflowManager(),
	}
}

func (im *IterationManager) SetWorkflow(workflow *Workflow) {
	im.workflowManager.SetWorkflow(*workflow)
}

// Run handles the iteration loop, either in streaming or non-streaming mode.
// Returns the final response message that can be passed to the next agent.
func (im *IterationManager) Run(ctx context.Context, msg LLMMessage) (LLMMessage, error) {
	im.logger.Log(fmt.Sprintf("Running iteration manager for agent %s with message:\n%v", im.agent.Name(), msg))

	iteration := 0

	for iteration < im.agent.IterationLimit() {
		switch msg.Role {
		case "user":
			im.agent.AddUserMessage(msg.Content)
		case "assistant":
			im.agent.AddAssistantMessage(msg.Content)
		}

		var iterationResponse []LLMMessage
		var err error

		im.logger.Log(fmt.Sprintf("======%s======\n", "[Messages]"))
		for _, message := range im.agent.Params().Messages {
			im.logger.Log(fmt.Sprintf("Message: %s", message))
		}
		im.logger.Log(fmt.Sprintf("======%s======\n", "[/Messages]"))
		if im.agent.Streaming() {
			iterationResponse, err = im.handleStreamingIteration(ctx)
		} else {
			iterationResponse, err = im.handleNonStreamingIteration(ctx)
		}

		if err != nil {
			return LLMMessage{}, err
		}

		composed := make([]string, 0)

		for _, response := range iterationResponse {
			im.logger.Log(fmt.Sprintf("======\n%s\n======\n", response.Content))
			composed = append(composed, response.Content)
			im.logger.Log(fmt.Sprintf("======\n%s\n======\n", response.Content))
		}

		msg = LLMMessage{
			Role:    "assistant",
			Content: strings.Join(composed, "\n\n"),
		}

		iteration++
	}

	return msg, nil
}

func (im *IterationManager) handleNonStreamingIteration(ctx context.Context) ([]LLMMessage, error) {
	im.hub.Add(hub.NewStatus(im.agent.Name(), "agent", "thinking"))

	res := im.agent.LLM().GenerateResponse(ctx, *im.agent.Params())

	if res.Error != nil {
		im.hub.Add(hub.NewError(im.agent.Name(), "agent", "error", res.Error.Error()))
		return nil, res.Error
	}

	out := make([]LLMMessage, 0)
	im.hub.Add(hub.NewMessage(im.agent.Name(), res.Content))

	out = append(out, LLMMessage{
		Role:    "assistant",
		Content: res.Content,
	})

	out = append(out, im.handleToolCalls(ctx, res.ToolCalls)...)

	return out, nil
}

func (im *IterationManager) handleStreamingIteration(ctx context.Context) ([]LLMMessage, error) {
	im.hub.Add(hub.NewStatus(im.agent.Name(), "agent", "thinking"))

	var (
		streamedResponse strings.Builder
		toolCalls        []ToolCall
		out              []LLMMessage
	)

	for chunk := range im.agent.LLM().StreamResponse(ctx, *im.agent.Params()) {
		if chunk.Error != nil {
			im.hub.Add(hub.NewError(im.agent.Name(), "agent", "error", chunk.Error.Error()))
		}

		switch chunk.Type {
		case ResponseTypeToolCall:
			toolCalls = chunk.ToolCalls
			out = im.handleToolCalls(ctx, toolCalls)
		case ResponseTypeContent:
			content := im.responseProcessor.ProcessChunkContent(chunk.Content)

			if content != "" {
				im.hub.Add(hub.NewChunk(im.agent.Name(), content))
				streamedResponse.WriteString(content)
			}
		}
	}

	out = append(out, LLMMessage{
		Role:    "assistant",
		Content: streamedResponse.String(),
	})

	out = append(out, im.handleToolCalls(ctx, toolCalls)...)

	return out, nil
}

func (im *IterationManager) handleToolCalls(ctx context.Context, toolCalls []ToolCall) []LLMMessage {
	out := make([]LLMMessage, 0)

	if len(toolCalls) > 0 {

		for _, toolCall := range toolCalls {
			im.hub.Add(hub.NewToolCall(im.agent.Name(), toolCall.Name, ""))
			toolResults, err := im.agent.GetTool(toolCall.Name).Execute(ctx, toolCall.Args)

			if err != nil {
				im.hub.Add(hub.NewError(im.agent.Name(), "agent", "error", err.Error()))
				return nil
			}

			out = append(out, LLMMessage{
				Role:    "assistant",
				Content: toolResults.(string),
			})
		}
	}

	return out
}
