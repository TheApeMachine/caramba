package provider

import (
	"github.com/gofiber/fiber/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/theapemachine/caramba/pkg/errnie"
	prvdrTools "github.com/theapemachine/caramba/pkg/provider/tools"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
OpenAIProvider implements an LLM provider that connects to OpenAI's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type OpenAIProvider struct {
	client *openai.Client
}

type OpenAIProviderOption func(*OpenAIProvider)

/*
NewOpenAIProvider creates a new OpenAI provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the OPENAI_API_KEY environment variable.
This can also be used for local AI, since most will follow the OpenAI API format.
*/
func NewOpenAIProvider(opts ...OpenAIProviderOption) *OpenAIProvider {
	prvdr := &OpenAIProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OpenAIProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) openai.ChatCompletionNewParams {
	var (
		params = openai.ChatCompletionNewParams{
			Model:            openai.ChatModel(tweaker.GetModel(tweaker.GetProvider())),
			Temperature:      openai.Float(tweaker.GetTemperature()),
			TopP:             openai.Float(tweaker.GetTopP()),
			FrequencyPenalty: openai.Float(tweaker.GetFrequencyPenalty()),
			PresencePenalty:  openai.Float(tweaker.GetPresencePenalty()),
			Tools:            tools.NewRegistry().ToOpenAI(),
		}
	)

	for _, msg := range request.Params.History {
		switch msg.Role.String() {
		case "system":
			params.Messages = append(params.Messages, openai.SystemMessage(msg.String()))
		case "user", "developer":
			params.Messages = append(params.Messages, openai.UserMessage(msg.String()))
		case "assistant":
			params.Messages = append(params.Messages, openai.AssistantMessage(msg.String()))
		case "tool":
			params.Messages = append(params.Messages, openai.ToolMessage(msg.String(), request.Params.ID))
		}
	}

	return params
}

func (prvdr *OpenAIProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	errnie.Trace("OpenAIProvider.Generate", "request", request)

	out := make(chan *task.TaskResponse)
	reqCtx := ctx.Context()

	go func() {
		defer close(out)

		var (
			params     = prvdr.prepare(ctx, request)
			completion *openai.ChatCompletion
			outTask    = request.Params
			err        error
		)

		if completion, err = prvdr.client.Chat.Completions.New(
			reqCtx,
			params,
		); errnie.Error(err) != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		outTask.AddMessage(task.NewAssistantMessage(
			completion.Choices[0].Message.Content,
		))

		if len(completion.Choices[0].Message.ToolCalls) == 0 {
			outTask.Status.State = task.TaskStateCompleted
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
			return
		}

		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			outTask.AddMessage(tools.NewRegistry().CallOpenAI(ctx, toolCall))
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *OpenAIProvider) Stream(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	errnie.Trace("OpenAIProvider.Stream", "request", request)

	out := make(chan *task.TaskResponse)
	reqCtx := ctx.Context()

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepare(ctx, request)
			outTask = request.Params
		)

		outTask.Status.State = task.TaskStateWorking

		stream := prvdr.client.Chat.Completions.NewStreaming(reqCtx, params)
		acc := openai.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			prvdr.handleChunk(ctx, acc, outTask, chunk, out)
		}

		if err := stream.Err(); errnie.Error(err) != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}
	}()

	return out, nil
}

func (prvdr *OpenAIProvider) handleChunk(
	ctx fiber.Ctx,
	acc openai.ChatCompletionAccumulator,
	outTask task.Task,
	chunk openai.ChatCompletionChunk,
	out chan *task.TaskResponse,
) {
	if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
		errnie.Debug("Received content chunk", "content", chunk.Choices[0].Delta.Content)
		outTask.AddMessage(task.NewAssistantMessage(chunk.Choices[0].Delta.Content))
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
		return
	}

	// Only send completion notification if we actually finished content
	if content, ok := acc.JustFinishedContent(); ok && content != "" {
		errnie.Debug("Accumulator detected end of content stream")
		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
		return
	}

	if refusal, ok := acc.JustFinishedRefusal(); ok {
		errnie.Warn("Assistant stream finished with refusal", "refusal", refusal)
		outTask.AddMessage(task.NewAssistantMessage("[Refused to answer]"))
		outTask.Status.State = task.TaskStateFailed
		out <- task.NewTaskResponse(task.WithResponseError(
			errnie.New(errnie.WithMessage("Assistant refused to answer")),
		))
		return
	}

	if tool, ok := acc.JustFinishedToolCall(); ok {
		errnie.Debug("Accumulator detected tool call completion")
		handler := prvdrTools.NewToolCallHandler(&mcp.CallToolRequest{
			Params: struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments,omitempty"`
				Meta      *struct {
					ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
				} `json:"_meta,omitempty"`
			}{
				Name: tool.Name,
				Arguments: map[string]interface{}{
					"arguments": tool.Arguments,
				},
			},
		})

		result, err := handler.Handle(ctx)

		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		var content string

		for _, c := range result.Content {
			switch c := c.(type) {
			case mcp.TextContent:
				content += c.Text
			}
		}

		if content != "" {
			outTask.Status.State = task.TaskStateCompleted
			outTask.AddMessage(task.NewAssistantMessage(content))
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
		}
		return
	}
}

func WithOpenAIAPIKey(apiKey string) OpenAIProviderOption {
	return func(prvdr *OpenAIProvider) {
		client := openai.NewClient(
			option.WithAPIKey(apiKey),
		)

		prvdr.client = &client
	}
}

type OpenAIEmbedder struct {
	client *openai.Client
}

type OpenAIEmbedderOption func(*OpenAIEmbedder)

func NewOpenAIEmbedder(opts ...OpenAIEmbedderOption) *OpenAIEmbedder {
	prvdr := &OpenAIEmbedder{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OpenAIEmbedder) Embed(
	ctx fiber.Ctx, request *task.TaskRequest,
) ([]float64, error) {
	var (
		outTask = request.Params
		input   string
	)

	// Get the input text from the last user message
	for i := len(outTask.History) - 1; i >= 0; i-- {
		if outTask.History[i].Role.String() == "user" {
			input = outTask.History[i].String()
			break
		}
	}

	if input == "" {
		outTask.Status.State = task.TaskStateFailed
		return nil, errnie.New(errnie.WithMessage("no input text found for embedding"))
	}

	embeddings, err := prvdr.client.Embeddings.New(ctx.Context(), openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(openai.EmbeddingModelTextEmbedding3Large),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: param.Opt[string]{Value: input},
		},
	})

	if errnie.Error(err) != nil {
		outTask.Status.State = task.TaskStateFailed
		return nil, errnie.New(errnie.WithError(err))
	}

	outTask.Status.State = task.TaskStateCompleted
	return embeddings.Data[0].Embedding, nil
}

func WithOpenAIEmbedderAPIKey(apiKey string) OpenAIEmbedderOption {
	return func(prvdr *OpenAIEmbedder) {
		client := openai.NewClient(
			option.WithAPIKey(apiKey),
		)

		prvdr.client = &client
	}
}
