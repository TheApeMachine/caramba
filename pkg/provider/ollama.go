package provider

import (
	"github.com/gofiber/fiber/v3"
	"github.com/ollama/ollama/api"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	client *api.Client
}

type OllamaProviderOption func(*OllamaProvider)

/*
NewOllamaProvider creates a new Ollama provider with the given host endpoint.
If host is empty, it will try to read from configuration.
*/
func NewOllamaProvider(opts ...OllamaProviderOption) *OllamaProvider {
	prvdr := &OllamaProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *OllamaProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) *api.ChatRequest {
	return &api.ChatRequest{
		Model: tweaker.GetModel(tweaker.GetProvider()),
		Options: map[string]any{
			"temperature": tweaker.GetTemperature(),
			"top_p":       tweaker.GetTopP(),
			"max_tokens":  tweaker.GetMaxTokens(),
		},
	}
}

func (prvdr *OllamaProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			outTask = request.Params
		)

		composed := &api.ChatRequest{
			Model: tweaker.GetModel(tweaker.GetProvider()),
			Options: map[string]any{
				"temperature": tweaker.GetTemperature(),
				"top_p":       tweaker.GetTopP(),
				"max_tokens":  tweaker.GetMaxTokens(),
			},
		}

		prvdr.client.Chat(ctx.Context(), composed, func(response api.ChatResponse) error {
			outTask.AddMessage(task.NewAssistantMessage(response.Message.Content))
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
			return nil
		})
	}()

	return out, nil
}

func (prvdr *OllamaProvider) Stream(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepare(ctx, request)
			outTask = request.Params
		)

		prvdr.client.Chat(ctx.Context(), params, func(response api.ChatResponse) error {
			outTask.AddMessage(task.NewAssistantMessage(response.Message.Content))
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
			return nil
		})
	}()

	return out, nil
}

func WithOllamaAPIKey(apiKey string) OllamaProviderOption {
	return func(prvdr *OllamaProvider) {
		prvdr.client = api.NewClient(nil, nil)
	}
}
