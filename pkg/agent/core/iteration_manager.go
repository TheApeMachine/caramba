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
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/process"
)

/*
IterationManager handles agent iteration logic, managing the conversation flow
between agents and LLMs. It handles message preparation, tool calls, streaming,
and message truncation to stay within token limits.
*/
type IterationManager struct {
	logger    *output.Logger
	hub       *hub.Queue
	agent     Agent
	breaking  bool
	iteration int
	limit     int
	out       []LLMMessage
}

/*
NewIterationManager creates a new IterationManager instance for handling
agent iterations. It initializes the manager with the provided agent and
default configuration settings.
*/
func NewIterationManager(agent Agent) *IterationManager {
	return &IterationManager{
		logger:    output.NewLogger(),
		hub:       hub.NewQueue(),
		agent:     agent,
		breaking:  false,
		iteration: 0,
		limit:     viper.GetViper().GetInt("settings.global.iteration_limit"),
	}
}

/*
Run handles the main iteration loop, either in streaming or non-streaming mode.
It manages the conversation flow, processes messages, and handles any errors that occur.
Returns the final set of messages and any error encountered.
*/
func (im *IterationManager) Run(ctx context.Context, msgs []LLMMessage) ([]LLMMessage, error) {
	if len(msgs) == 0 {
		return msgs, im.logError(errors.New("message is empty"))
	}

	// Reset the output messages.
	im.out = make([]LLMMessage, 0)

	iteration := 0

	for !im.breaking || iteration < min(im.agent.IterationLimit(), im.limit) {
		im.prepareMessages(msgs)
		im.agent.SetStatus(AgentStatusThinking)

		var err error

		if im.agent.Streaming() {
			msgs, err = im.streamingIteration(ctx, msgs)
		} else {
			msgs, err = im.nonStreamingIteration(ctx, msgs)
		}

		if err != nil {
			return msgs, err
		}

		iteration++
	}

	return im.out, nil
}

/*
prepareMessages prepares the message array for the LLM by adding the system prompt
and truncating messages to stay within token limits. It also logs the prepared messages
for debugging purposes.
*/
func (im *IterationManager) prepareMessages(msgs []LLMMessage) {
	im.agent.Params().Messages = append([]LLMMessage{
		{Role: "system", Content: im.agent.SystemPrompt()},
	}, msgs...)

	im.agent.Params().Messages = im.truncateMessages(im.agent.Params().Messages)
	im.logger.Log(im.agent.Name(), spew.Sdump(im.agent.Params().Messages))
}

/*
nonStreamingIteration performs a single non-streaming iteration where the entire
response is generated at once. It processes the response, handles any tool calls,
and returns the updated message array and any errors.
*/
func (im *IterationManager) nonStreamingIteration(ctx context.Context, msgs []LLMMessage) ([]LLMMessage, error) {
	res := im.agent.LLM().GenerateResponse(ctx, *im.agent.Params())

	if res.Error != nil {
		return msgs, im.logError(res.Error)
	}

	if im.agent.Params().Schema != nil {
		im.out = append(im.out, im.createAgentMessage(res.Content))
	}

	im.hub.Add(&hub.Event{
		Origin:  im.agent.Name(),
		Topic:   hub.TopicTypeMessage,
		Type:    hub.EventTypeResponse,
		Message: res.Content,
	})

	msgs = append(msgs, im.createAgentMessage(res.Content))

	processResults := im.handleProcess(ctx, res.Content)
	toolResults := im.handleToolCalls(ctx, res.ToolCalls)

	return append(append(msgs, processResults...), toolResults...), nil
}

/*
streamingIteration performs a single streaming iteration where the response
is received in chunks. It handles the streamed content and tool calls, assembling
the complete response and returning the updated message array.
*/
func (im *IterationManager) streamingIteration(ctx context.Context, msgs []LLMMessage) ([]LLMMessage, error) {
	var response strings.Builder
	var toolCalls []ToolCall

	for chunk := range im.agent.LLM().StreamResponse(ctx, *im.agent.Params()) {
		if chunk.Error != nil {
			im.logError(chunk.Error)
			continue
		}

		switch chunk.Type {
		case ResponseTypeToolCall:
			toolCalls = chunk.ToolCalls
		case ResponseTypeContent:
			if chunk.Content != "" {
				im.hub.Add(&hub.Event{
					Origin:  im.agent.Name(),
					Topic:   hub.TopicTypeMessage,
					Type:    hub.EventTypeChunk,
					Message: chunk.Content,
				})
				response.WriteString(chunk.Content)
			}
		}
	}

	if im.agent.Params().Schema == nil {
		im.out = append(im.out, im.createAgentMessage(response.String()))
	}

	msgs = append(msgs, im.createAgentMessage(response.String()))
	return append(msgs, im.handleToolCalls(ctx, toolCalls)...), nil
}

/*
createAgentMessage creates a new LLMMessage with the assistant role and prefixes
the content with the agent's name for identification purposes.
*/
func (im *IterationManager) createAgentMessage(content string) LLMMessage {
	return LLMMessage{
		Role:    "assistant",
		Content: "ADDED BY: " + im.agent.Name() + "\n\n" + content,
	}
}

/*
handleToolCalls processes an array of tool calls, executing each tool and
collecting the results. It adds the results to the message array and handles
any system commands or errors that occur during execution.
*/
func (im *IterationManager) handleToolCalls(ctx context.Context, toolCalls []ToolCall) []LLMMessage {
	var results []LLMMessage

	for _, call := range toolCalls {
		im.hub.Add(&hub.Event{
			Origin:  im.agent.Name(),
			Topic:   hub.TopicTypeMessage,
			Type:    hub.EventTypeToolCall,
			Message: fmt.Sprintf("%v", call.Args),
		})
		result, err := im.agent.GetTool(call.Name).Execute(ctx, call.Args)

		if err != nil {
			im.logError(err)
			continue
		}

		if call.Name == "system" {
			im.breaking = true
			continue
		}

		results = append(results, LLMMessage{
			Role:    "assistant",
			Content: im.formatResult(result),
		})

		im.out = append(im.out, results...)
	}

	return results
}

/*
formatResult formats the result of a tool execution into a string representation.
It handles different result types, converting non-string results to JSON.
*/
func (im *IterationManager) formatResult(result interface{}) string {
	switch v := result.(type) {
	case string:
		return v
	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			im.logError(err)
			return "Error: Unable to process tool results"
		}
		return string(jsonBytes)
	}
}

/*
handleProcess processes responses that contain JSON-formatted process instructions.
It handles memory lookup and mutation operations, and returns the results as messages.
*/
func (im *IterationManager) handleProcess(ctx context.Context, response string) []LLMMessage {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(response), &data); err != nil {
		im.logError(err)
		return nil
	}

	name, ok := data["name"].(string)
	if !ok {
		return nil
	}

	var result string
	var err error

	switch name {
	case "memory_lookup":
		var proc process.MemoryLookup
		if err = json.Unmarshal([]byte(response), &proc); err == nil {
			result, err = im.agent.Memory().Query(ctx, &proc)
		}
	case "memory_mutate":
		var proc process.MemoryMutate
		if err = json.Unmarshal([]byte(response), &proc); err == nil {
			err = im.agent.Memory().Mutate(ctx, &proc)
			result = "Memory mutated successfully"
		}
	}

	if err != nil {
		im.logError(err)
		return nil
	}

	im.out = append(im.out, im.createAgentMessage(result))
	return []LLMMessage{{Role: "assistant", Content: result}}
}

/*
truncateMessages truncates the message array to stay within the maximum token limit
for LLM context. It keeps the system message, user message, and as many recent messages
as possible while staying under the token limit.
*/
func (im *IterationManager) truncateMessages(msgs []LLMMessage) []LLMMessage {
	const maxTokens = 7500 // 8000 - 500 buffer

	if len(msgs) < 2 {
		return msgs
	}

	result := msgs[:2]
	tokens := 0
	for _, msg := range result {
		tokens += im.estimateTokens(msg)
	}

	var recent []LLMMessage
	for i := len(msgs) - 1; i >= 2; i-- {
		msgTokens := im.estimateTokens(msgs[i])
		if tokens+msgTokens > maxTokens {
			break
		}
		recent = append(recent, msgs[i])
		tokens += msgTokens
	}

	for i := len(recent) - 1; i >= 0; i-- {
		result = append(result, recent[i])
	}

	return result
}

/*
estimateTokens estimates the number of tokens in a message using the tiktoken
library with the specified model encoding. It counts tokens for both the role
and content fields of the message.
*/
func (im *IterationManager) estimateTokens(msg LLMMessage) int {
	encoding, err := tiktoken.EncodingForModel("gpt-4o-mini")
	if err != nil {
		log.Error("Error getting encoding", "error", err)
		return 0
	}

	tokensPerMessage := 4
	return tokensPerMessage +
		len(encoding.Encode(msg.Role, nil, nil)) +
		len(encoding.Encode(msg.Content, nil, nil))
}

/*
logError logs an error using the logger and returns the same error.
This is a utility function for logging and propagating errors.
*/
func (im *IterationManager) logError(err error) error {
	im.logger.Error(im.agent.Name(), err)
	return err
}
