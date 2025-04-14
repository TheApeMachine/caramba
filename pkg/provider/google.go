package provider

import (
	"context"
	"errors"
	"fmt"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/utils"
	"google.golang.org/genai"
)

/*
GoogleProvider implements an LLM provider that connects to Google's Gemini API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type GoogleProvider struct {
	client *genai.Client
}

type GoogleProviderOption func(*GoogleProvider)

/*
NewGoogleProvider creates a new Google Gemini provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the GOOGLE_API_KEY environment variable.
*/
func NewGoogleProvider(opts ...GoogleProviderOption) *GoogleProvider {
	prvdr := &GoogleProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *GoogleProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) (*genai.GenerateContentConfig, []*genai.Content) {
	var (
		chatConfig = &genai.GenerateContentConfig{
			Temperature:      utils.Ptr(float32(tweaker.GetTemperature())),
			TopP:             utils.Ptr(float32(tweaker.GetTopP())),
			FrequencyPenalty: utils.Ptr(float32(tweaker.GetFrequencyPenalty())),
			PresencePenalty:  utils.Ptr(float32(tweaker.GetPresencePenalty())),
			Tools:            make([]*genai.Tool, 0),
		}
		messages = make([]*genai.Content, 0)
	)

	// Convert messages
	for _, msg := range request.Params.History {
		content := &genai.Content{
			Role:  msg.Role.String(),
			Parts: []*genai.Part{{Text: msg.String()}},
		}
		messages = append(messages, content)
	}

	// Add tools from registry
	for _, tool := range tools.NewRegistry().GetToolNames() {
		chatConfig.Tools = append(chatConfig.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: tool,
			}},
		})
	}

	return chatConfig, messages
}

func (prvdr *GoogleProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			chatConfig, messages = prvdr.prepare(ctx, request)
			outTask              = request.Params
		)

		resp, err := prvdr.client.Models.GenerateContent(
			ctx.Context(),
			tweaker.GetModel(tweaker.GetProvider()),
			messages,
			chatConfig,
		)

		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		if len(resp.Candidates) == 0 {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(errors.New("no response candidates")))
			return
		}

		contentText := ""
		for _, part := range resp.Candidates[0].Content.Parts {
			contentText += part.Text
		}

		outTask.AddMessage(task.NewAssistantMessage(contentText))

		// Handle tool calls
		if resp.Candidates[0].Content.Parts[0].FunctionCall != nil {
			fc := resp.Candidates[0].Content.Parts[0].FunctionCall
			outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
				`{"name": "%s", "arguments": %v}`,
				fc.Name,
				fc.Args,
			), fc.Name))
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *GoogleProvider) Stream(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			chatConfig, messages = prvdr.prepare(ctx, request)
			outTask              = request.Params
		)

		outTask.Status.State = task.TaskStateWorking

		for response, err := range prvdr.client.Models.GenerateContentStream(
			ctx.Context(),
			tweaker.GetModel(tweaker.GetProvider()),
			messages,
			chatConfig,
		) {
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				errnie.New(errnie.WithError(err))
				continue
			}

			if len(response.Candidates) == 0 {
				continue
			}

			prvdr.handleChunk(ctx, response, outTask, out)
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *GoogleProvider) handleChunk(
	ctx fiber.Ctx,
	response *genai.GenerateContentResponse,
	outTask *task.Task,
	out chan *task.TaskResponse,
) {
	for _, part := range response.Candidates[0].Content.Parts {
		if part.Text != "" {
			outTask.AddMessage(task.NewAssistantMessage(part.Text))
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
		}

		if part.FunctionCall != nil {
			fc := part.FunctionCall

			outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
				`{"name": "%s", "arguments": %v}`,
				fc.Name,
				fc.Args,
			), fc.Name))

			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
		}
	}
}

func WithGoogleAPIKey(apiKey string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.client, _ = genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
	}
}
