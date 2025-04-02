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
	"github.com/anthropics/anthropic-sdk-go/packages/param"
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

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	prvdr := &AnthropicProvider{
		client: &client,
		buffer: stream.NewBuffer(),
		params: &Params{},
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *AnthropicProvider) ID() string {
	return "anthropic"
}

func (prvdr *AnthropicProvider) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.AnthropicProvider.Generate")

	var (
		out = make(chan *datura.Artifact)
		err error
	)

	go func() {
		defer close(out)

		select {
		case <-prvdr.ctx.Done():
			errnie.Debug("provider.AnthropicProvider.Generate.ctx.Done")
			prvdr.cancel()
			return
		case artifact := <-buffer:
			if err := artifact.To(prvdr.params); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			composed := anthropic.MessageNewParams{
				Model:       anthropic.Model(prvdr.params.Model),
				Temperature: anthropic.Float(prvdr.params.Temperature),
				TopP:        anthropic.Float(prvdr.params.TopP),
			}

			if prvdr.params.MaxTokens > 1 {
				composed.MaxTokens = int64(prvdr.params.MaxTokens)
			}

			if err = prvdr.buildMessages(&composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if err = prvdr.buildTools(&composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if prvdr.params.ResponseFormat != nil {
				if err = prvdr.buildResponseFormat(&composed); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			if prvdr.params.Stream {
				if err = prvdr.handleStreamingRequest(&composed); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			} else {
				if err = prvdr.handleSingleRequest(&composed); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			out <- datura.New(datura.WithPayload(prvdr.params.Marshal()))
		}
	}()

	return out
}

func (prvdr *AnthropicProvider) Name() string {
	return "anthropic"
}

type AnthropicProviderOption func(*AnthropicProvider)

func WithAnthropicAPIKey(apiKey string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		provider.client.Options = append(provider.client.Options, option.WithAPIKey(apiKey))
	}
}

func (prvdr *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	response, err := prvdr.client.Messages.New(prvdr.ctx, *params)
	if err != nil {
		return errnie.Error(err)
	}

	if response.Content == nil {
		err = errors.New("content is nil")
		return errnie.Error(err)
	}

	msg := &Message{
		Role: MessageRoleAssistant,
		Name: prvdr.params.Model,
	}

	for _, block := range response.Content {
		switch block := block.AsAny().(type) {
		case anthropic.TextBlock:
			msg.Content += block.Text
		case anthropic.ToolUseBlock:
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      block.Name,
					Arguments: block.JSON.Input.Raw(),
				},
			})
			errnie.Info("toolCall detected", "name", block.Name, "id", block.ID)
		}
	}

	prvdr.params.Messages = append(prvdr.params.Messages, msg)

	if _, err = io.Copy(prvdr.buffer, datura.New(
		datura.WithPayload(prvdr.params.Marshal()),
	)); err != nil {
		return errnie.Error(err)
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
		if err = accumulatedMessage.Accumulate(chunk); err != nil {
			return errnie.Error(err)
		}

		switch event := chunk.AsAny().(type) {
		case anthropic.ContentBlockStartEvent:
			if event.ContentBlock.Name != "" {
				if _, err = io.Copy(prvdr.buffer, datura.New(
					datura.WithPayload([]byte(event.ContentBlock.Name+": ")),
				)); errnie.Error(err) != nil {
					continue
				}
			}
		case anthropic.ContentBlockDeltaEvent:
			if event.Delta.Text != "" {
				if _, err = io.Copy(prvdr.buffer, datura.New(
					datura.WithPayload([]byte(event.Delta.Text)),
				)); errnie.Error(err) != nil {
					continue
				}
			}
			if event.Delta.PartialJSON != "" {
				if _, err = io.Copy(prvdr.buffer, datura.New(
					datura.WithPayload([]byte(event.Delta.PartialJSON)),
				)); errnie.Error(err) != nil {
					continue
				}
			}
		case anthropic.ContentBlockStopEvent:
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte("\n\n")),
			)); errnie.Error(err) != nil {
				continue
			}
		case anthropic.MessageStopEvent:
			if _, err = io.Copy(prvdr.buffer, datura.New(
				datura.WithPayload([]byte("\n")),
			)); errnie.Error(err) != nil {
				continue
			}
		}

		// Handle tool calls if present in the accumulated message
		if len(accumulatedMessage.Content) > 0 {
			for _, block := range accumulatedMessage.Content {
				if block.Type == "tool_use" {
					toolData, err := json.Marshal(block)
					if err != nil {
						errnie.Error("failed to marshal tool_use block", "error", err)
						continue
					}

					var toolInfo struct {
						ID    string                 `json:"id"`
						Name  string                 `json:"name"`
						Input map[string]interface{} `json:"input"`
					}

					if err := json.Unmarshal(toolData, &toolInfo); err != nil {
						errnie.Error("failed to unmarshal tool data", "error", err)
						continue
					}

					inputJSON, err := json.Marshal(toolInfo.Input)
					if err != nil {
						errnie.Error("failed to marshal tool input", "error", err)
						continue
					}

					msg := &Message{
						Role:    MessageRoleAssistant,
						Name:    prvdr.params.Model,
						Content: "",
						ToolCalls: []ToolCall{{
							ID:   toolInfo.ID,
							Type: "function",
							Function: ToolCallFunction{
								Name:      toolInfo.Name,
								Arguments: string(inputJSON),
							},
						}},
					}

					prvdr.params.Messages = append(prvdr.params.Messages, msg)
					errnie.Info("toolCall detected (streaming)", "name", toolInfo.Name)
				}
			}
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return err
	}

	return nil
}

func (prvdr *AnthropicProvider) buildMessages(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	msgParams := make([]anthropic.MessageParam, 0)

	// Keep track of which tool messages we've processed
	processedTools := make(map[int]bool)

	for i, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			messageParams.System = []anthropic.TextBlockParam{
				{Text: message.Content},
			}
		case "user":
			// First add the regular user message
			userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(message.Content))

			// Check if there's a tool message that follows this user message
			if i+1 < len(prvdr.params.Messages) && prvdr.params.Messages[i+1].Role == "tool" {
				// Mark the tool message as processed
				processedTools[i+1] = true

				// Now manually handle the tool result by adding it to the message in a custom way
				// The exact mechanism depends on how the Anthropic SDK handles tool results
				// This would typically involve creating a message with both text and tool results
				// Since we don't have exact structure, we'll annotate the user message

				// Add a note about tool result in the user message
				toolMsg := prvdr.params.Messages[i+1]
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

	messageParams.Messages = msgParams
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
			if len(property.Enum) > 0 {
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
			Name:        tool.Function.Name,
			Description: param.NewOpt(tool.Function.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: properties,
			},
		}

		toolParams = append(toolParams, toolParam)
	}

	// Set the tools
	toolUnionParams := make([]anthropic.ToolUnionParam, 0, len(toolParams))
	for _, tool := range toolParams {
		toolUnionParams = append(toolUnionParams, anthropic.ToolUnionParam{
			OfTool: &tool,
		})
	}
	messageParams.Tools = toolUnionParams
	return nil
}

func (prvdr *AnthropicProvider) buildResponseFormat(
	messageParams *anthropic.MessageNewParams,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params.ResponseFormat == nil {
		return nil
	}

	messageParams.Messages = append(
		messageParams.Messages,
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
	client *anthropic.Client
	params *Params
	ctx    context.Context
	cancel context.CancelFunc
}

func NewAnthropicEmbedder() *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	return &AnthropicEmbedder{
		client: &client,
		params: &Params{},
		ctx:    ctx,
		cancel: cancel,
	}
}

func (embedder *AnthropicEmbedder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.AnthropicEmbedder.Generate")
	errnie.Warn("provider.AnthropicEmbedder.Generate not implemented")

	out := make(chan *datura.Artifact)
	close(out)
	return out
}

type AnthropicEmbedderOption func(*AnthropicEmbedder)

func WithAnthropicEmbedderAPIKey(apiKey string) AnthropicEmbedderOption {
	return func(embedder *AnthropicEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithAnthropicEmbedderEndpoint(endpoint string) AnthropicEmbedderOption {
	return func(embedder *AnthropicEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithBaseURL(endpoint))
	}
}
