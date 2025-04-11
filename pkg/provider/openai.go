package provider

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client *openai.Client
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(opts ...OpenAIProviderOption) *OpenAIProvider {
	errnie.Trace("provider.NewOpenAIProvider")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// Return an error if the API key is missing
		errnie.Error(errors.New("OPENAI_API_KEY environment variable not set"))
		return nil
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &OpenAIProvider{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OpenAIProvider) Generate(
	params ProviderParams,
) (<-chan ProviderEvent, error) {
	errnie.Info("provider.Generate", "supplier", "openai", "stream", params.Stream)

	composed := &openai.ChatCompletionNewParams{
		Model:            openai.ChatModel(params.Model),
		Temperature:      openai.Float(params.Temperature),
		TopP:             openai.Float(params.TopP),
		FrequencyPenalty: openai.Float(params.FrequencyPenalty),
		PresencePenalty:  openai.Float(params.PresencePenalty),
	}

	if params.MaxTokens > 1 {
		composed.MaxTokens = openai.Int(int64(params.MaxTokens))
	}

	var err error

	if err = prvdr.buildMessages(composed, params.Messages); err != nil {
		return nil, err
	}

	if err = prvdr.buildTools(composed, params.Tools); err != nil {
		return nil, err
	}

	if params.ResponseFormat != (ResponseFormat{}) {
		if err = prvdr.buildResponseFormat(composed, params.ResponseFormat); err != nil {
			return nil, err
		}
	}

	if params.Stream {
		return prvdr.handleStreamingRequest(composed)
	} else {
		eventChan := make(chan ProviderEvent, 1)
		go func() {
			defer close(eventChan)
			event, err := prvdr.handleSingleRequest(composed)
			if err != nil {
				errnie.Error("Error in handleSingleRequest goroutine", "error", err)
			} else {
				eventChan <- event
			}
		}()
		return eventChan, nil
	}
}

func (prvdr *OpenAIProvider) Name() string {
	return "openai"
}

type OpenAIProviderOption func(*OpenAIProvider)

func WithAPIKey(apiKey string) OpenAIProviderOption {
	return func(prvdr *OpenAIProvider) {
		prvdr.client.Options = append(prvdr.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithEndpoint(endpoint string) OpenAIProviderOption {
	return func(prvdr *OpenAIProvider) {
		prvdr.client.Options = append(prvdr.client.Options, option.WithBaseURL(endpoint))
	}
}

/*
handleSingleRequest processes a single (non-streaming) completion request
*/
func (prvdr *OpenAIProvider) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) (ProviderEvent, error) {
	errnie.Trace("provider.handleSingleRequest")

	var (
		err        error
		completion *openai.ChatCompletion
	)

	errnie.Info("provider.handleSingleRequest", "tools", params.Tools)

	if completion, err = prvdr.client.Chat.Completions.New(
		prvdr.ctx, *params,
	); errnie.Error(err) != nil {
		return ProviderEvent{}, err
	}

	// Use a temporary list for pending calls
	pendingCalls := []PendingToolCall{}
	assistantContent := completion.Choices[0].Message.Content // Store potential text content

	// Process tool calls if any
	if len(completion.Choices[0].Message.ToolCalls) > 0 {
		assistantContent = "" // OpenAI often leaves content empty when making tool calls
		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			errnie.Info("ToolCall requested by LLM", "tool", toolCall.Function.Name, "id", toolCall.ID)

			// Parse arguments from JSON
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
				errnie.Error("failed to parse tool call arguments", "error", err, "id", toolCall.ID)
				// Decide how to handle parse errors - skip this call, return error?
				continue // Skipping for now
			}

			pendingCalls = append(pendingCalls, PendingToolCall{
				ID: toolCall.ID, // Store the crucial ID
				Request: mcp.CallToolRequest{
					Params: struct {
						Name      string                 `json:"name"`
						Arguments map[string]interface{} `json:"arguments,omitempty"`
						Meta      *struct {
							ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
						} `json:"_meta,omitempty"`
					}{
						Name:      toolCall.Function.Name,
						Arguments: args,
					},
				},
			})
		}
	}

	// Return ProviderEvent with assistant message content and pending tool calls
	return ProviderEvent{
		Message: Message{
			Role:    "assistant",
			Name:    params.Model,
			Content: assistantContent, // Will be empty if tool calls were made
			// ToolCalls field in Message struct is for *requests* made by assistant,
			// which we are packaging into ProviderEvent.ToolCalls instead.
		},
		ToolCalls: pendingCalls, // Pass the list with IDs
	}, nil
}

/*
handleStreamingRequest processes a streaming completion request
and sends ProviderEvents on the returned channel.
*/
func (prvdr *OpenAIProvider) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (<-chan ProviderEvent, error) {
	errnie.Trace("provider.handleStreamingRequest starting")

	events := make(chan ProviderEvent, 10) // Buffered channel for events

	// Start the OpenAI stream
	// Note: NewStreaming itself doesn't return an error here
	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)

	// Start processing the stream in a goroutine
	go func() {
		defer close(events)
		defer stream.Close() // Ensure stream is closed when goroutine exits
		errnie.Trace("provider.handleStreamingRequest goroutine started")

		// Accumulator helps manage tool calls across chunks
		acc := openai.ChatCompletionAccumulator{}
		var accumulatedContent strings.Builder // Use strings.Builder

		for stream.Next() {
			chunk := stream.Current()
			errnie.Debug("Received stream chunk") // Less verbose default logging

			// Use accumulator to process chunks
			if ok := acc.AddChunk(chunk); !ok {
				errnie.Error("Accumulator failed to add chunk", "chunk", chunk)
				continue
			}

			// --- Send Content Chunks ---
			if len(chunk.Choices) > 0 {
				handleContentChunk(chunk.Choices[0], params.Model, events, &accumulatedContent)
			}

			// --- Send Completed Tool Calls ---
			if toolCall, ok := acc.JustFinishedToolCall(); ok {
				handleFinishedToolCall(toolCall, params.Model, events, &accumulatedContent)
			}

			// Handle JustFinishedContent() and JustFinishedRefusal() if needed for streaming
			// These accumulator states might be useful for more complex streaming logic,
			// but are not strictly required for the current implementation.
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			errnie.Error("OpenAI stream error", "error", err)
			// Send a specific event to signal the error, using the Message field.
			events <- ProviderEvent{
				Message: Message{
					Role:    "error",
					Content: err.Error(),
				},
			}
		}

		errnie.Trace("provider.handleStreamingRequest goroutine finished")
	}()

	errnie.Trace("provider.handleStreamingRequest returning channel")
	return events, nil // Return the channel immediately
}

// handleContentChunk processes a content delta chunk and sends an event.
func handleContentChunk(
	choice openai.ChatCompletionChunkChoice,
	model string,
	events chan<- ProviderEvent,
	accumulatedContent *strings.Builder,
) {
	if choice.Delta.Content != "" {
		contentPart := choice.Delta.Content
		accumulatedContent.WriteString(contentPart)
		events <- ProviderEvent{
			Message: Message{
				Role:    "assistant",
				Name:    model,
				Content: contentPart, // Send just the delta
			},
			// Artifact streaming flags (Index, Append, LastChunk) seem to be handled
			// at the task handler level (e.g., task_send_subscribe.go) rather than
			// needing to be part of the provider event itself.
		}
	}
}

// handleFinishedToolCall processes a completed tool call from the stream and sends an event.
func handleFinishedToolCall(
	toolCall openai.FinishedChatCompletionToolCall,
	model string,
	events chan<- ProviderEvent,
	accumulatedContent *strings.Builder,
) {
	errnie.Info("Accumulator finished tool call", "tool", toolCall.Name, "id", toolCall.Id)

	// Parse arguments
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(toolCall.Arguments), &args); err != nil {
		errnie.Error("failed to parse tool call arguments in stream", "error", err, "id", toolCall.Id)
		return // Skip sending event if parsing fails
	}

	// Send event with the completed tool call request
	events <- ProviderEvent{
		ToolCalls: []PendingToolCall{
			{
				ID: toolCall.Id,
				Request: mcp.CallToolRequest{
					Params: struct {
						Name      string                 `json:"name"`
						Arguments map[string]interface{} `json:"arguments,omitempty"`
						Meta      *struct {
							ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
						} `json:"_meta,omitempty"`
					}{
						Name:      toolCall.Name,
						Arguments: args,
					},
				},
			},
		},
		// Message content might be empty here if only tool calls are made
		Message: Message{
			Role: "assistant",
			Name: model,
			// Content: accumulatedContent.String(), // Consider if sending full accumulated content here is needed
		},
	}
	accumulatedContent.Reset() // Reset accumulated content after sending tool call
}

/*
buildMessages converts ContextData messages to OpenAI API format
*/
func (prvdr *OpenAIProvider) buildMessages(
	composed *openai.ChatCompletionNewParams,
	messages []Message,
) (err error) {
	errnie.Trace("provider.buildMessages")

	openaiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(msg.Content))
		case "user":
			openaiMessages = append(openaiMessages, openai.UserMessage(msg.Content))
		case "assistant":
			toolCalls := msg.ToolCalls

			tcs := make([]openai.ChatCompletionMessageToolCallParam, 0, len(toolCalls))

			for idx, toolCall := range msg.ToolCalls {
				// NOTE: The OpenAI API requires an ID for historical tool calls made by the assistant.
				// Our current mcp.CallToolRequest structure doesn't store the original ID assigned by
				// the LLM when the call was *first* made. We generate a placeholder here.
				// This might need refinement if precise historical correlation becomes necessary.
				id := fmt.Sprintf("call_%d_%s", idx, toolCall.Params.Name) // Placeholder ID
				name := toolCall.Params.Name

				// Serialize arguments back to JSON string for the API
				var argumentsStr string
				if argsBytes, err := json.Marshal(toolCall.Params.Arguments); err == nil {
					argumentsStr = string(argsBytes)
				} else {
					// Handle error: maybe log, maybe set a default, maybe skip?
					// For now, setting to empty JSON object string and logging.
					errnie.Warn("Failed to marshal tool arguments for OpenAI history", "error", err, "tool", name)
					argumentsStr = "{}"
				}

				tcs = append(tcs, openai.ChatCompletionMessageToolCallParam{
					ID:   id,
					Type: "function", // Use the string literal "function"
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      name,
						Arguments: argumentsStr, // Pass the JSON string
					},
				})
			}

			assistantMsg := openai.AssistantMessage(msg.Content)
			if len(tcs) > 0 {
				assistantMsg = openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(msg.Content),
						},
						ToolCalls: tcs,
						Role:      "assistant",
					},
				}
			}

			openaiMessages = append(openaiMessages, assistantMsg)
		case "tool":
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(msg.Content),
					},
					ToolCallID: msg.ID, // Use the Message ID field for ToolCallID
					Role:       "tool",
				},
			})
		default:
			return errnie.Error(
				errors.New("unknown message role"),
				"role", msg.Role,
			)
		}
	}

	composed.Messages = openaiMessages
	return nil
}

/*
buildTools takes the tools from the generic params and converts them to OpenAI API format.
It is important to return nil early when there are no tools, because passing an empty array
to the OpenAI API will cause strange behavior, like the model guessing random tools.
*/
func (prvdr *OpenAIProvider) buildTools(
	openaiParams *openai.ChatCompletionNewParams,
	tools []mcp.Tool,
) (err error) {
	errnie.Trace("provider.buildTools", "tools", tools)

	if len(tools) == 0 {
		// No tools, no shoes, no dice.
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(tools))

	for _, tool := range tools {
		toolParam := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters: openai.FunctionParameters{
					"type":       tool.InputSchema.Type,
					"properties": tool.InputSchema.Properties,
				},
			},
		}
		toolsOut = append(toolsOut, toolParam)
	}

	if len(toolsOut) > 0 {
		openaiParams.Tools = toolsOut
	}

	return nil
}

/*
buildResponseFormat converts the response format from the generic params to OpenAI API format.
This will force the model to use structured output, and return a JSON object.
Setting Strict to true will make sure the only thing returned is the JSON object.
If you want this to be combined with the ability to call tools, you can set Strict to false.
*/
func (prvdr *OpenAIProvider) buildResponseFormat(
	openaiParams *openai.ChatCompletionNewParams,
	format ResponseFormat,
) (err error) {
	errnie.Trace("provider.buildResponseFormat")

	openaiParams.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        format.Name,
				Description: param.NewOpt(format.Description),
				Schema:      format.Schema,
				Strict:      param.NewOpt(format.Strict),
			},
		},
	}

	return nil
}

/*
OpenAIEmbedder implements an LLM provider that connects to OpenAI's API.
It supports embedding generation for text inputs.
*/
type OpenAIEmbedder struct {
	client *openai.Client
	ctx    context.Context
	cancel context.CancelFunc
}

type OpenAIEmbedderOption func(*OpenAIEmbedder)

/*
NewOpenAIEmbedder creates a new OpenAI embedder with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIEmbedder(opts ...OpenAIEmbedderOption) *OpenAIEmbedder {
	errnie.Trace("provider.NewOpenAIEmbedder")

	apiKey := os.Getenv("OPENAI_API_KEY")
	ctx, cancel := context.WithCancel(context.Background())

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	embedder := &OpenAIEmbedder{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, opt := range opts {
		opt(embedder)
	}

	return embedder
}

/*
Generate implements the Generator interface for OpenAIEmbedder.
It takes input text through a channel and returns embeddings through another channel.
*/
func (embedder *OpenAIEmbedder) Generate(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("provider.OpenAIEmbedder.Generate")

	content, err := artifact.DecryptPayload()
	if err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if len(content) == 0 {
		return datura.New(datura.WithError(errnie.Error(errors.New("content is empty"))))
	}

	response, err := embedder.client.Embeddings.New(embedder.ctx, openai.EmbeddingNewParams{
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{string(content)}},
		Model:          openai.EmbeddingModelTextEmbeddingAda002,
		Dimensions:     openai.Int(tweaker.GetQdrantDimension()),
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return datura.New(datura.WithError(errnie.Error(err)))
	}

	if len(response.Data) == 0 {
		return datura.New(datura.WithError(errnie.Error(errors.New("no embeddings returned"))))
	}

	// Convert float64 embeddings to float32
	embeddings := response.Data[0].Embedding
	float32Embeddings := make([]float32, len(embeddings))
	for i, v := range embeddings {
		float32Embeddings[i] = float32(v)
	}

	// Convert embeddings to bytes
	embeddingsBytes := make([]byte, len(float32Embeddings)*4)
	for i, v := range float32Embeddings {
		binary.LittleEndian.PutUint32(embeddingsBytes[i*4:], math.Float32bits(v))
	}

	return datura.New(datura.WithEncryptedPayload(embeddingsBytes))
}

func WithOpenAIEmbedderAPIKey(apiKey string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithAPIKey(apiKey))
	}
}

func WithOpenAIEmbedderEndpoint(endpoint string) OpenAIEmbedderOption {
	return func(embedder *OpenAIEmbedder) {
		embedder.client.Options = append(embedder.client.Options, option.WithBaseURL(endpoint))
	}
}
