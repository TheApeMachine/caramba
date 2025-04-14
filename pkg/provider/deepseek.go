package provider

import (
	"encoding/json"
	"fmt"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
DeepseekProvider implements an LLM provider that connects to Deepseek's API.
It supports regular chat completions and streaming responses.
*/
type DeepseekProvider struct {
	client *deepseek.Client
}

type DeepseekProviderOption func(*DeepseekProvider)

/*
NewDeepseekProvider creates a new Deepseek provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the DEEPSEEK_API_KEY environment variable.
*/
func NewDeepseekProvider(opts ...DeepseekProviderOption) *DeepseekProvider {
	prvdr := &DeepseekProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *DeepseekProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) *deepseek.ChatCompletionRequest {
	params := &deepseek.ChatCompletionRequest{
		Model:            tweaker.GetModel(tweaker.GetProvider()),
		Temperature:      float32(tweaker.GetTemperature()),
		TopP:             float32(tweaker.GetTopP()),
		FrequencyPenalty: float32(tweaker.GetFrequencyPenalty()),
		PresencePenalty:  float32(tweaker.GetPresencePenalty()),
	}

	for _, msg := range request.Params.History {
		switch msg.Role.String() {
		case "system":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: msg.String(),
			})
		case "user":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: msg.String(),
			})
		case "assistant":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: msg.String(),
			})
		case "tool":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleTool,
				Content: msg.String(),
			})
		}
	}

	for _, toolName := range tools.NewRegistry().GetToolNames() {
		params.Tools = append(params.Tools, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name: toolName,
				Parameters: &deepseek.FunctionParameters{
					Type: "object",
				},
			},
		})
	}

	return params
}

func (prvdr *DeepseekProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (*deepseek.Message, error) {
	params := prvdr.prepare(ctx, request)

	resp, err := prvdr.client.CreateChatCompletion(ctx.Context(), params)

	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	if len(resp.Choices) == 0 {
		return nil, errnie.New(errnie.WithMessage("no choices in response"))
	}

	choice := resp.Choices[0]
	if choice.Message.ToolCalls != nil && len(choice.Message.ToolCalls) > 0 {
		return &choice.Message, nil
	}

	return &choice.Message, nil
}

func (prvdr *DeepseekProvider) prepareStream(
	ctx fiber.Ctx, request *task.TaskRequest,
) *deepseek.StreamChatCompletionRequest {
	params := &deepseek.StreamChatCompletionRequest{
		Model:            tweaker.GetModel(tweaker.GetProvider()),
		Temperature:      float32(tweaker.GetTemperature()),
		TopP:             float32(tweaker.GetTopP()),
		FrequencyPenalty: float32(tweaker.GetFrequencyPenalty()),
		PresencePenalty:  float32(tweaker.GetPresencePenalty()),
	}

	for _, msg := range request.Params.History {
		switch msg.Role.String() {
		case "system":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleSystem,
				Content: msg.String(),
			})
		case "user":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleUser,
				Content: msg.String(),
			})
		case "assistant":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleAssistant,
				Content: msg.String(),
			})
		case "tool":
			params.Messages = append(params.Messages, deepseek.ChatCompletionMessage{
				Role:    deepseek.ChatMessageRoleTool,
				Content: msg.String(),
			})
		}
	}

	return params
}

func (prvdr *DeepseekProvider) Stream(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepareStream(ctx, request)
			outTask = request.Params
		)

		outTask.Status.State = task.TaskStateWorking

		stream, err := prvdr.client.CreateChatCompletionStream(
			ctx.Context(), params,
		)

		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		defer stream.Close()

		for {
			response, err := stream.Recv()

			if err != nil {
				if err.Error() == "EOF" {
					break
				}

				errnie.New(errnie.WithError(err))
				continue
			}

			prvdr.handleChunk(response, outTask, out)
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *DeepseekProvider) handleChunk(
	chunk *deepseek.StreamChatCompletionResponse,
	outTask task.Task,
	out chan *task.TaskResponse,
) {
	if len(chunk.Choices) == 0 {
		return
	}

	if content := chunk.Choices[0].Delta.Content; content != "" {
		outTask.AddMessage(task.NewAssistantMessage(content))
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}

	for _, toolCall := range chunk.Choices[0].Delta.ToolCalls {
		toolJSON, err := json.Marshal(toolCall)

		if err != nil {
			errnie.New(errnie.WithError(err))
			return
		}

		outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
			`{"name": "%s", "id": "tool-0", "arguments": %s}`,
			toolCall.Function.Name,
			string(toolJSON),
		), toolCall.Function.Name))

		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}
}

func WithDeepseekAPIKey(apiKey string) DeepseekProviderOption {
	return func(prvdr *DeepseekProvider) {
		prvdr.client = deepseek.NewClient(apiKey)
	}
}
