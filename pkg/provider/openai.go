package provider

import (
	"github.com/gofiber/fiber/v3"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params     = prvdr.prepare(ctx, request)
			completion *openai.ChatCompletion
			outTask    = request.Params
			err        error
		)

		if completion, err = prvdr.client.Chat.Completions.New(
			ctx.Context(), params,
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
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params     = prvdr.prepare(ctx, request)
			outTask    = request.Params
			completion *openai.ChatCompletion
		)

		outTask.Status.State = task.TaskStateWorking

		stream := prvdr.client.Chat.Completions.NewStreaming(ctx.Context(), params)
		acc := openai.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			prvdr.handleChunk(ctx, acc, outTask, completion, chunk, out)
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
	completion *openai.ChatCompletion,
	chunk openai.ChatCompletionChunk,
	out chan *task.TaskResponse,
) {
	if _, ok := acc.JustFinishedContent(); ok {
		outTask.AddMessage(task.NewAssistantMessage(
			completion.Choices[0].Message.Content,
		))

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}

	if refusal, ok := acc.JustFinishedRefusal(); ok {
		println()
		println("finish-event: refusal stream finished:", refusal)
		println()
	}

	if tool, ok := acc.JustFinishedToolCall(); ok {
		if len(completion.Choices[0].Message.ToolCalls) == 0 {
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
			return
		}

		outTask.AddMessage(tools.NewRegistry().CallOpenAITool(ctx, tool))
	}

	if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
		outTask.AddMessage(task.NewAssistantMessage(
			completion.Choices[0].Message.Content,
		))

		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
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
