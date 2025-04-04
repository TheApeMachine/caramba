package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/mark3labs/mcp-go/mcp"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"

	"capnproto.org/go/capnp/v3"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client  *anthropic.Client
	pctx    context.Context
	ctx     context.Context
	cancel  context.CancelFunc
	segment *capnp.Segment
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
		pctx:   ctx,
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
	params params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.Artifact {
	model, err := params.Model()

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		composed := anthropic.MessageNewParams{
			Model:       anthropic.Model(model),
			Temperature: anthropic.Float(params.Temperature()),
			TopP:        anthropic.Float(params.TopP()),
		}

		if params.MaxTokens() > 1 {
			composed.MaxTokens = int64(params.MaxTokens())
		}

		if err = prvdr.buildMessages(&composed, ctx); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildTools(&composed, tools); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		format, err := params.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(&composed, format); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if params.Stream() {
			prvdr.handleStreamingRequest(&composed, out)
		} else {
			prvdr.handleSingleRequest(&composed, out)
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

func WithAnthropicEndpoint(endpoint string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		provider.client.Options = append(provider.client.Options, option.WithBaseURL(endpoint))
	}
}

func (prvdr *AnthropicProvider) handleSingleRequest(
	params *anthropic.MessageNewParams,
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleSingleRequest")

	response, err := prvdr.client.Messages.New(prvdr.ctx, *params)
	if err != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if response.Content == nil {
		channel <- datura.New(datura.WithError(errnie.Error(errors.New("content is nil"))))
		return
	}

	// Create a new message using Cap'n Proto
	msg, err := aicontext.NewMessage(prvdr.segment)
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Set message fields
	if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	if err = msg.SetName(string(params.Model)); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	var content string
	var toolCalls []anthropic.ToolUseBlock

	for _, block := range response.Content {
		switch block := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += block.Text
		case anthropic.ToolUseBlock:
			toolCalls = append(toolCalls, block)
		}
	}

	if err = msg.SetContent(content); errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	// Abort early if there are no tool calls
	if len(toolCalls) == 0 {
		channel <- datura.New(datura.WithPayload([]byte(content)))
		return
	}

	// Create tool calls list
	toolCallList, err := msg.NewToolCalls(int32(len(toolCalls)))
	if errnie.Error(err) != nil {
		channel <- datura.New(datura.WithError(errnie.Error(err)))
		return
	}

	for i, toolCall := range toolCalls {
		errnie.Info("toolCall", "tool", toolCall.Name, "id", toolCall.ID)

		if err = toolCallList.At(i).SetId(toolCall.ID); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = toolCallList.At(i).SetName(toolCall.Name); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		args := toolCall.JSON.Input.Raw()
		if err = toolCallList.At(i).SetArguments(args); errnie.Error(err) != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
	}

	// Create artifact with message content
	channel <- datura.New(datura.WithPayload([]byte(content)))
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (prvdr *AnthropicProvider) handleStreamingRequest(
	params *anthropic.MessageNewParams,
	channel chan *datura.Artifact,
) {
	errnie.Debug("provider.handleStreamingRequest")

	stream := prvdr.client.Messages.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	accumulatedMessage := anthropic.Message{}

	for stream.Next() {
		chunk := stream.Current()
		if err := accumulatedMessage.Accumulate(chunk); err != nil {
			channel <- datura.New(datura.WithError(errnie.Error(err)))
			continue
		}

		switch event := chunk.AsAny().(type) {
		case anthropic.ContentBlockStartEvent:
			if event.ContentBlock.Name != "" {
				channel <- datura.New(
					datura.WithRole(datura.ArtifactRoleAnswer),
					datura.WithScope(datura.ArtifactScopeGeneration),
					datura.WithPayload([]byte(event.ContentBlock.Name+": ")),
				)
			}
		case anthropic.ContentBlockDeltaEvent:
			if event.Delta.Text != "" {
				channel <- datura.New(
					datura.WithRole(datura.ArtifactRoleAnswer),
					datura.WithScope(datura.ArtifactScopeGeneration),
					datura.WithPayload([]byte(event.Delta.Text)),
				)
			}
			if event.Delta.PartialJSON != "" {
				channel <- datura.New(
					datura.WithRole(datura.ArtifactRoleAnswer),
					datura.WithScope(datura.ArtifactScopeGeneration),
					datura.WithPayload([]byte(event.Delta.PartialJSON)),
				)
			}
		case anthropic.ContentBlockStopEvent:
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte("\n\n")),
			)
		case anthropic.MessageStopEvent:
			channel <- datura.New(
				datura.WithRole(datura.ArtifactRoleAnswer),
				datura.WithScope(datura.ArtifactScopeGeneration),
				datura.WithPayload([]byte("\n")),
			)
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

					// Create a new message using Cap'n Proto
					msg, err := aicontext.NewMessage(prvdr.segment)
					if errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					// Set message fields
					if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					if err = msg.SetName(string(params.Model)); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					if err = msg.SetContent(""); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					// Create tool calls list
					toolCallList, err := msg.NewToolCalls(1)
					if errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					if err = toolCallList.At(0).SetId(toolInfo.ID); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					if err = toolCallList.At(0).SetName(toolInfo.Name); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					if err = toolCallList.At(0).SetArguments(string(inputJSON)); errnie.Error(err) != nil {
						channel <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					errnie.Info("toolCall detected (streaming)", "name", toolInfo.Name)
					channel <- datura.New(
						datura.WithRole(datura.ArtifactRoleAnswer),
						datura.WithScope(datura.ArtifactScopeGeneration),
						datura.WithPayload(inputJSON),
					)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		channel <- datura.New(
			datura.WithError(errnie.Error("Streaming error", "error", err)),
		)
	}
}

func (prvdr *AnthropicProvider) buildMessages(
	messageParams *anthropic.MessageNewParams,
	ctx aicontext.Context,
) (err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()

	if errnie.Error(err) != nil {
		return err
	}

	msgParams := make([]anthropic.MessageParam, 0, msgs.Len())
	var systemMessage string

	for i := 0; i < msgs.Len(); i++ {
		msg := msgs.At(i)

		role, err := msg.Role()
		if errnie.Error(err) != nil {
			return err
		}

		content, err := msg.Content()
		if errnie.Error(err) != nil {
			return err
		}

		switch role {
		case "system":
			systemMessage = content
		case "user":
			msgParams = append(msgParams, anthropic.NewUserMessage(anthropic.NewTextBlock(content)))
		case "assistant":
			toolCalls, err := msg.ToolCalls()
			if errnie.Error(err) != nil {
				return err
			}

			if toolCalls.Len() > 0 {
				// Create assistant message with text content
				msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(content)))

				// Add tool calls information
				for j := 0; j < toolCalls.Len(); j++ {
					toolCall := toolCalls.At(j)

					name, err := toolCall.Name()
					if errnie.Error(err) != nil {
						return err
					}

					args, err := toolCall.Arguments()
					if errnie.Error(err) != nil {
						return err
					}

					toolNote := fmt.Sprintf("[Tool Call: %s, Arguments: %s]", name, args)
					msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(toolNote)))
				}
			} else {
				// Regular assistant message without tool calls
				msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(content)))
			}
		case "tool":
			id, err := msg.Id()
			if errnie.Error(err) != nil {
				return err
			}

			// Create a tool result message
			toolMsg := anthropic.NewUserMessage(
				anthropic.NewTextBlock(fmt.Sprintf("[Tool Result from %s: %s]", id, content)),
			)

			msgParams = append(msgParams, toolMsg)
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}

	// Set system message if present
	if systemMessage != "" {
		messageParams.System = []anthropic.TextBlockParam{
			{Text: systemMessage},
		}
	}

	messageParams.Messages = msgParams
	return nil
}

func (prvdr *AnthropicProvider) buildTools(
	messageParams *anthropic.MessageNewParams,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	// If no tools, skip
	if len(tools) == 0 {
		return nil
	}

	// Prepare the tools
	toolParams := make([]anthropic.ToolParam, 0, len(tools))

	for _, tool := range tools {
		// Create a tool parameter with this schema
		toolParam := anthropic.ToolParam{
			Name:        tool.Name,
			Description: param.NewOpt(tool.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: tool.InputSchema.Properties,
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
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	name, err := format.Name()
	if errnie.Error(err) != nil {
		return err
	}

	description, err := format.Description()
	if errnie.Error(err) != nil {
		return err
	}

	schema, err := format.Schema()
	if errnie.Error(err) != nil {
		return err
	}

	// If no format specified, skip
	if name == "" && description == "" && schema == "" {
		return nil
	}

	messageParams.Messages = append(
		messageParams.Messages,
		anthropic.NewAssistantMessage(
			anthropic.NewTextBlock(
				strings.Join([]string{
					"Format your response as a JSON object using the following schema.",
					fmt.Sprintf("Schema:\n\n%v", schema),
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
	ctx    context.Context
	cancel context.CancelFunc
}

func NewAnthropicEmbedder(opts ...AnthropicEmbedderOption) *AnthropicEmbedder {
	errnie.Debug("provider.NewAnthropicEmbedder")

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	embedder := &AnthropicEmbedder{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
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
