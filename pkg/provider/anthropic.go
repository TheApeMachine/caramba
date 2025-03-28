package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client *anthropic.Client
	buffer *stream.Buffer
	params *Params
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewAnthropicProvider creates a new Anthropic provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the ANTHROPIC_API_KEY environment variable.
*/
func NewAnthropicProvider(opts ...AnthropicProviderOption) *AnthropicProvider {
	errnie.Debug("provider.NewAnthropicProvider")

	apiKey := os.Getenv("ANTHROPIC_API_KEY")

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	prvdr := &AnthropicProvider{
		client: anthropic.NewClient(
			option.WithAPIKey(apiKey),
		),
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			errnie.Debug("provider.AnthropicProvider.buffer.fn")
			return errnie.Error(artfct.To(params))
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

type AnthropicProviderOption func(*AnthropicProvider)

func WithAnthropicAPIKey(apiKey string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		provider.client = anthropic.NewClient(
			option.WithAPIKey(apiKey),
		)
	}
}

/*
Read implements the io.Reader interface.
*/
func (prvdr *AnthropicProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *AnthropicProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.AnthropicProvider.Write")

	if n, err = prvdr.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	composed := anthropic.MessageNewParams{
		Model:       anthropic.F(prvdr.params.Model),
		Temperature: anthropic.F(prvdr.params.Temperature),
		TopP:        anthropic.F(prvdr.params.TopP),
		MaxTokens:   anthropic.F(int64(prvdr.params.MaxTokens)),
	}

	prvdr.buildMessages(&composed)
	prvdr.buildTools(&composed)
	prvdr.buildResponseFormat(&composed)

	if prvdr.params.Stream {
		prvdr.handleStreamingRequest(&composed)
	} else {
		prvdr.handleSingleRequest(&composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *AnthropicProvider) Close() error {
	errnie.Debug("provider.AnthropicProvider.Close")
	return prvdr.buffer.Close()
}

func (prvdr *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	response, err := prvdr.client.Messages.New(prvdr.ctx, *params)
	if errnie.Error(err) != nil {
		return
	}

	if response.Content == nil {
		errnie.Error("failed to get message content", "error", "content is nil")
		return errors.New("content is nil")
	}

	msg := &Message{
		Role: MessageRoleAssistant,
		Name: prvdr.params.Model,
	}

	prvdr.params.Messages = append(prvdr.params.Messages, msg)

	for _, block := range response.Content {
		switch block.Type {
		case anthropic.ContentBlockTypeText:
			msg.Content += block.Text
		case anthropic.ContentBlockTypeToolUse:
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      block.Name,
					Arguments: block.JSON.Input.Raw(),
				},
			})
		}
	}

	if _, err = io.Copy(prvdr.buffer, datura.New(
		datura.WithPayload(prvdr.params.Marshal()),
	)); errnie.Error(err) != nil {
		return err
	}

	return nil
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := prvdr.client.Messages.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	accumulatedMessage := anthropic.Message{}

	for stream.Next() {
		chunk := stream.Current()
		accumulatedMessage.Accumulate(chunk)

		// Extract text content from deltas
		var content string
		switch delta := chunk.Delta.(type) {
		case anthropic.ContentBlockDeltaEventDelta:
			content = delta.Text
		}

		if content == "" {
			continue
		}

		if _, err = io.Copy(prvdr.buffer, datura.New(
			datura.WithPayload([]byte(content)),
		)); errnie.Error(err) != nil {
			continue
		}
	}

	// After streaming is done, check for tool calls in the complete message
	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return errnie.Error(err)
	}

	// Check if we have tool calls in the accumulated message
	hasToolCalls := false
	for _, block := range accumulatedMessage.Content {
		if block.Type == "tool_use" {
			hasToolCalls = true
			break
		}
	}

	// If we have tool calls, process them
	if hasToolCalls {
		// Create a message with the tool calls
		msg := &Message{
			Role:      MessageRoleAssistant,
			Name:      prvdr.params.Model,
			Content:   "", // Content was already streamed
			ToolCalls: make([]ToolCall, 0),
		}

		// Extract tool calls
		for _, block := range accumulatedMessage.Content {
			if block.Type == "tool_use" {
				// Try to extract tool information using JSON
				toolData, err := json.Marshal(block)
				if err != nil {
					errnie.Error("failed to marshal tool_use block", "error", err)
					continue
				}

				// Parse the tool use data
				var toolInfo struct {
					ID    string                 `json:"id"`
					Name  string                 `json:"name"`
					Input map[string]interface{} `json:"input"`
				}

				if err := json.Unmarshal(toolData, &toolInfo); err != nil {
					errnie.Error("failed to unmarshal tool data", "error", err)
					continue
				}

				// Convert input to JSON string
				inputJSON, err := json.Marshal(toolInfo.Input)
				if err != nil {
					errnie.Error("failed to marshal tool input", "error", err)
					continue
				}

				// Add to tool calls
				msg.ToolCalls = append(msg.ToolCalls, ToolCall{
					ID:   toolInfo.ID,
					Type: "function",
					Function: ToolCallFunction{
						Name:      toolInfo.Name,
						Arguments: string(inputJSON),
					},
				})

				errnie.Info("toolCall detected (streaming)", "name", toolInfo.Name)
			}
		}

		// Add to messages and send through buffer if we extracted tools
		if len(msg.ToolCalls) > 0 {
			prvdr.params.Messages = append(prvdr.params.Messages, msg)

			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload(prvdr.params.Marshal()),
			)); errnie.Error(err) != nil {
				return err
			}
		}
	}

	return nil
}

func (p *AnthropicProvider) buildMessages(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if p.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	msgParams := make([]anthropic.MessageParam, 0)

	// Keep track of which tool messages we've processed
	processedTools := make(map[int]bool)

	for i, message := range p.params.Messages {
		switch message.Role {
		case "system":
			messageParams.System = anthropic.F([]anthropic.TextBlockParam{
				anthropic.NewTextBlock(message.Content),
			})
		case "user":
			// First add the regular user message
			userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content))

			// Check if there's a tool message that follows this user message
			if i+1 < len(p.params.Messages) && p.params.Messages[i+1].Role == "tool" {
				// Mark the tool message as processed
				processedTools[i+1] = true

				// Now manually handle the tool result by adding it to the message in a custom way
				// The exact mechanism depends on how the Anthropic SDK handles tool results
				// This would typically involve creating a message with both text and tool results
				// Since we don't have exact structure, we'll annotate the user message

				// Add a note about tool result in the user message
				toolMsg := p.params.Messages[i+1]
				userMsg = anthropic.NewUserMessage(
					anthropic.NewTextBlock(fmt.Sprintf("%s\n\n[Tool Result from %s: %s]",
						message.Content, toolMsg.Reference, toolMsg.Content)),
				)
			}

			msgParams = append(msgParams, userMsg)
		case "assistant":
			// Handle regular assistant message or one with tool calls
			if len(message.ToolCalls) > 0 {
				// For messages with tool calls, we need to add both the text content
				// and information about the tool calls in a custom way

				// Create the text message first
				textMsg := anthropic.NewAssistantMessage(
					anthropic.NewTextBlock(message.Content),
				)

				// Add the message
				msgParams = append(msgParams, textMsg)

				// For models that support tool calling, we would add tool call information here
				// Since we don't have exact format, we add the tool call info in text
				for _, toolCall := range message.ToolCalls {
					// Add a note about tool calls
					toolNote := fmt.Sprintf("[Tool Call: %s, Arguments: %s]",
						toolCall.Function.Name, toolCall.Function.Arguments)

					toolMsg := anthropic.NewAssistantMessage(
						anthropic.NewTextBlock(toolNote),
					)

					msgParams = append(msgParams, toolMsg)
				}
			} else {
				// Regular assistant message without tool calls
				msgParams = append(msgParams, anthropic.NewAssistantMessage(
					anthropic.NewTextBlock(message.Content),
				))
			}
		case "tool":
			// Tool messages are handled when processing user messages
			// If this tool message wasn't processed with a user message, process it now
			if !processedTools[i] {
				// Create a tool result message
				toolMsg := anthropic.NewUserMessage(
					anthropic.NewTextBlock(fmt.Sprintf("[Tool Result from %s: %s]",
						message.Reference, message.Content)),
				)

				msgParams = append(msgParams, toolMsg)
			}
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	messageParams.Messages = anthropic.F(msgParams)
	return nil
}

func (prvdr *AnthropicProvider) buildTools(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildTools")

	// If no tools, skip
	if len(prvdr.params.Tools) == 0 {
		return nil
	}

	// Prepare the tools
	toolParams := make([]anthropic.ToolParam, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		// Create properties map for schema
		properties := make(map[string]interface{})

		// Add each property
		for _, property := range tool.Function.Parameters.Properties {
			propDef := map[string]interface{}{
				"type":        property.Type,
				"description": property.Description,
			}

			// Only add enum if not empty
			if property.Enum != nil && len(property.Enum) > 0 {
				propDef["enum"] = property.Enum
			}

			properties[property.Name] = propDef
		}

		// Create overall schema
		schema := map[string]interface{}{
			"type":       "object",
			"properties": properties,
		}

		// Add required fields if present
		if len(tool.Function.Parameters.Required) > 0 {
			schema["required"] = tool.Function.Parameters.Required
		}

		// Create a tool parameter with this schema
		toolParam := anthropic.ToolParam{
			Name:        anthropic.F(tool.Function.Name),
			Description: anthropic.F(tool.Function.Description),
			InputSchema: anthropic.Raw[interface{}](schema),
		}

		toolParams = append(toolParams, toolParam)
	}

	// Set the tools
	messageParams.Tools = anthropic.F(toolParams)
	return nil
}

func (prvdr *AnthropicProvider) buildResponseFormat(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params.ResponseFormat == nil {
		return nil
	}

	messageParams.Messages.Value = append(
		messageParams.Messages.Value,
		anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(
				strings.Join([]string{
					"Format your response as a JSON object using the following schema.",
					fmt.Sprintf("Schema:\n\n%v", prvdr.params.ResponseFormat.Schema),
					"Strictly follow the schema. Do not leave out required fields, and do not include any non-existent fields or properties.",
					"Output only the JSON object, nothing else, and no Markdown code block.",
				}, "\n\n"),
			),
		),
	)

	return nil
}

type AnthropicEmbedder struct {
	params   *Params
	apiKey   string
	endpoint string
	client   *anthropic.Client
}

func NewAnthropicEmbedder(apiKey string, endpoint string) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	return &AnthropicEmbedder{
		params:   &Params{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   anthropic.NewClient(option.WithAPIKey(apiKey)),
	}
}

func (embedder *AnthropicEmbedder) Read(p []byte) (n int, err error) {
	errnie.Warn("provider.AnthropicEmbedder.Read not implemented")
	return 0, nil
}

func (embedder *AnthropicEmbedder) Write(p []byte) (n int, err error) {
	errnie.Warn("provider.AnthropicEmbedder.Write not implemented")
	return len(p), nil
}

func (embedder *AnthropicEmbedder) Close() error {
	errnie.Warn("provider.AnthropicEmbedder.Close not implemented")

	embedder.params = nil
	return nil
}
