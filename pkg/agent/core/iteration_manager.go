package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/davecgh/go-spew/spew"
	"github.com/pkoukk/tiktoken-go"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/process"
)

// IterationManager handles agent iteration logic.
type IterationManager struct {
	logger          *output.Logger
	hub             *hub.Queue
	agent           Agent
	workflowManager *WorkflowManager
}

// NewIterationManager creates a new IterationManager.
func NewIterationManager(
	agent Agent,
) *IterationManager {
	return &IterationManager{
		logger:          output.NewLogger(),
		hub:             hub.NewQueue(),
		agent:           agent,
		workflowManager: NewWorkflowManager(),
	}
}

func (im *IterationManager) SetWorkflow(workflow *Workflow) {
	im.workflowManager.SetWorkflow(*workflow)
}

// Run handles the iteration loop, either in streaming or non-streaming mode.
// Returns the final response message that can be passed to the next agent.
func (im *IterationManager) Run(ctx context.Context, msgs []LLMMessage) ([]LLMMessage, error) {
	if len(msgs) == 0 {
		return msgs, im.logger.Error(im.agent.Name(), errors.New("message is empty"))
	}

	// Overwrite the agent's messages, leaving only the first message,
	// which is the system prompt.
	im.agent.Params().Messages = append([]LLMMessage{
		{
			Role:    "system",
			Content: im.agent.SystemPrompt(),
		},
	}, msgs...)

	var err error

	if im.agent.Streaming() {
		msgs, err = im.handleStreamingIteration(ctx, msgs)
	} else {
		msgs, err = im.handleNonStreamingIteration(ctx, msgs)
	}

	if err != nil {
		im.logger.Error(im.agent.Name(), err)
		return msgs, err
	}

	return msgs, nil
}

func (im *IterationManager) handleNonStreamingIteration(
	ctx context.Context,
	msgs []LLMMessage,
) ([]LLMMessage, error) {
	im.hub.Add(hub.NewStatus(im.agent.Name(), "agent", "thinking"))

	im.agent.Params().Messages = im.truncateMessages(im.agent.Params().Messages)
	im.logger.Log(im.agent.Name(), spew.Sdump(im.agent.Params().Messages))

	res := im.agent.LLM().GenerateResponse(ctx, *im.agent.Params())

	if res.Error != nil {
		im.logger.Error(im.agent.Name(), res.Error)
		return msgs, res.Error
	}

	im.hub.Add(hub.NewMessage(im.agent.Name(), res.Content))

	msgs = append(msgs, LLMMessage{
		Role:    "assistant",
		Content: "ADDED BY: " + im.agent.Name() + "\n\n" + res.Content,
	})

	msgs = append(msgs, im.handleProcess(ctx, res.Content)...)
	msgs = append(msgs, im.handleToolCalls(ctx, res.ToolCalls)...)

	return msgs, nil
}

func (im *IterationManager) handleStreamingIteration(
	ctx context.Context,
	msgs []LLMMessage,
) ([]LLMMessage, error) {
	im.hub.Add(hub.NewStatus(im.agent.Name(), "agent", "thinking"))

	var (
		streamedResponse strings.Builder
		toolCalls        []ToolCall
	)

	im.agent.Params().Messages = im.truncateMessages(im.agent.Params().Messages)
	im.logger.Log(im.agent.Name(), spew.Sdump(im.agent.Params().Messages))

	for chunk := range im.agent.LLM().StreamResponse(ctx, *im.agent.Params()) {
		if chunk.Error != nil {
			im.logger.Error(im.agent.Name(), chunk.Error)
		}

		switch chunk.Type {
		case ResponseTypeToolCall:
			toolCalls = chunk.ToolCalls
			msgs = append(msgs, im.handleToolCalls(ctx, toolCalls)...)
		case ResponseTypeContent:
			if chunk.Content != "" {
				im.hub.Add(hub.NewChunk(
					im.agent.Name(),
					chunk.Content,
				))
				streamedResponse.WriteString(chunk.Content)
			}
		}
	}

	msgs = append(msgs, LLMMessage{
		Role:    "assistant",
		Content: "ADDED BY: " + im.agent.Name() + "\n\n" + streamedResponse.String(),
	})

	msgs = append(msgs, im.handleToolCalls(ctx, toolCalls)...)

	return msgs, nil
}

func (im *IterationManager) handleToolCalls(ctx context.Context, toolCalls []ToolCall) []LLMMessage {
	out := make([]LLMMessage, 0)

	if len(toolCalls) > 0 {
		for _, toolCall := range toolCalls {
			im.hub.Add(hub.NewToolCall(im.agent.Name(), toolCall.Name, fmt.Sprintf("%v", toolCall.Args)))
			toolResults, err := im.agent.GetTool(toolCall.Name).Execute(ctx, toolCall.Args)

			if err != nil {
				im.logger.Error(im.agent.Name(), err)
				return nil
			}

			// Handle different return types from tool.Execute
			var content string
			switch v := toolResults.(type) {
			case string:
				content = v
			default:
				// For any other type, try JSON marshal
				jsonBytes, err := json.Marshal(v)
				if err != nil {
					im.logger.Error(im.agent.Name(), err)
					content = "Error: Unable to process tool results"
					break
				}

				content = string(jsonBytes)
			}

			out = append(out, LLMMessage{
				Role:    "assistant",
				Content: content,
			})
		}
	}

	return out
}

func (im *IterationManager) handleProcess(ctx context.Context, response string) []LLMMessage {
	var buf map[string]any

	err := json.Unmarshal([]byte(response), &buf)

	if err != nil {
		im.logger.Error(im.agent.Name(), err)
		return nil
	}

	switch buf["name"].(string) {
	case "memory_lookup":
		proc := &process.MemoryLookup{}
		err = json.Unmarshal([]byte(response), proc)

		if err != nil {
			im.logger.Error(im.agent.Name(), err)
			return nil
		}

		results, err := im.agent.Memory().Query(ctx, proc)

		if err != nil {
			im.logger.Error(im.agent.Name(), err)
			return nil
		}

		return []LLMMessage{
			{
				Role:    "assistant",
				Content: results,
			},
		}
	case "memory_mutate":
		proc := &process.MemoryMutate{}
		err = json.Unmarshal([]byte(response), proc)

		if err != nil {
			im.logger.Error(im.agent.Name(), err)
			return nil
		}

		err = im.agent.Memory().Mutate(ctx, proc)

		if err != nil {
			im.logger.Error(im.agent.Name(), err)
			return nil
		}

		return []LLMMessage{
			{
				Role:    "assistant",
				Content: "Memory mutated successfully",
			},
		}
	}

	return nil
}

func (im *IterationManager) truncateMessages(msgs []LLMMessage) []LLMMessage {
	// Always include first two messages (system prompt and user message)
	if len(msgs) < 2 {
		return msgs
	}

	maxTokens := 8000 - 500 // Reserve tokens for response
	totalTokens := 0

	out := make([]LLMMessage, 0)

	// Always include the system prompt and user message
	for _, msg := range msgs[:2] {
		out = append(out, msg)
		totalTokens += EstimateTokens(map[string]string{"role": msg.Role, "content": msg.Content})
	}

	// Truncate the rest of the messages, keeping the most recent ones, by running through the messages
	// in reverse order.
	for i := len(msgs) - 1; i >= 2; i-- {
		msg := msgs[i]
		totalTokens += EstimateTokens(map[string]string{"role": msg.Role, "content": msg.Content})

		if totalTokens > maxTokens {
			break
		}

		out = append(out, msg)
	}

	return out
}

func EstimateTokens(msg map[string]string) int {
	encoding, err := tiktoken.EncodingForModel("gpt-4o-mini")

	if err != nil {
		log.Error("Error getting encoding", "error", err)
		return 0
	}

	tokensPerMessage := 4 // As per OpenAI's token estimation guidelines

	numTokens := tokensPerMessage
	numTokens += len(encoding.Encode(msg["role"], nil, nil))
	numTokens += len(encoding.Encode(msg["content"], nil, nil))

	return numTokens
}
