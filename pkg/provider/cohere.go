package provider

import (
	"encoding/json"
	"fmt"
	"io"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
CohereProvider implements an LLM provider that connects to Cohere's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type CohereProvider struct {
	client *cohereclient.Client
}

type CohereProviderOption func(*CohereProvider)

/*
NewCohereProvider creates a new Cohere provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the COHERE_API_KEY environment variable.
*/
func NewCohereProvider(opts ...CohereProviderOption) *CohereProvider {
	prvdr := &CohereProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *CohereProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) *cohere.ChatStreamRequest {
	params := &cohere.ChatStreamRequest{
		Model:            cohere.String(tweaker.GetModel(tweaker.GetProvider())),
		Temperature:      cohere.Float64(tweaker.GetTemperature()),
		P:                cohere.Float64(tweaker.GetTopP()),
		FrequencyPenalty: cohere.Float64(tweaker.GetFrequencyPenalty()),
		PresencePenalty:  cohere.Float64(tweaker.GetPresencePenalty()),
	}

	// Convert messages
	messageList := make([]*cohere.Message, 0)
	var systemMessage string

	for _, msg := range request.Params.History {
		switch msg.Role.String() {
		case "system":
			systemMessage = msg.String()
		case "user":
			messageList = append(messageList, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: msg.String(),
				},
			})
		case "assistant":
			messageList = append(messageList, &cohere.Message{
				Role: "chatbot",
				Chatbot: &cohere.ChatMessage{
					Message: msg.String(),
				},
			})
		case "tool":
			messageList = append(messageList, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: fmt.Sprintf("[Tool Result: %s]", msg.String()),
				},
			})
		}
	}

	if systemMessage != "" {
		params.Preamble = cohere.String(systemMessage)
	}

	params.ChatHistory = messageList

	// Add tools from registry
	toolList := make([]*cohere.Tool, 0)
	for _, toolName := range tools.NewRegistry().GetToolNames() {
		toolList = append(toolList, &cohere.Tool{
			Name: toolName,
			ParameterDefinitions: map[string]*cohere.ToolParameterDefinitionsValue{
				"arguments": {
					Type:     "object",
					Required: cohere.Bool(true),
				},
			},
		})
	}
	params.Tools = toolList

	return params
}

func (prvdr *CohereProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepare(ctx, request)
			outTask = request.Params
		)

		// Convert stream request to regular chat request
		chatRequest := &cohere.ChatRequest{
			Model:       params.Model,
			Message:     params.Message,
			ChatHistory: params.ChatHistory,
			Preamble:    params.Preamble,
			Tools:       params.Tools,
			Temperature: params.Temperature,
		}

		response, err := prvdr.client.Chat(ctx.Context(), chatRequest)
		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		outTask.AddMessage(task.NewAssistantMessage(response.Text))

		// Handle tool calls
		for i, toolCall := range response.GetToolCalls() {
			name := toolCall.GetName()
			id := fmt.Sprintf("tool-%d", i)

			paramBytes, err := json.Marshal(toolCall.GetParameters())
			if err != nil {
				errnie.New(errnie.WithError(err))
				continue
			}

			outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
				`{"name": "%s", "id": "%s", "arguments": %s}`,
				name,
				id,
				string(paramBytes),
			), name))
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *CohereProvider) Stream(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepare(ctx, request)
			outTask = request.Params
		)

		outTask.Status.State = task.TaskStateWorking

		stream, err := prvdr.client.ChatStream(ctx.Context(), params)
		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}
		defer stream.Close()

		for {
			chunk, err := stream.Recv()
			if err != nil {
				if err == io.EOF {
					break
				}
				errnie.New(errnie.WithError(err))
				continue
			}

			prvdr.handleChunk(chunk, outTask, out)
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *CohereProvider) handleChunk(
	chunk cohere.StreamedChatResponse,
	outTask *task.Task,
	out chan *task.TaskResponse,
) {
	if content := chunk.TextGeneration.String(); content != "" {
		outTask.AddMessage(task.NewAssistantMessage(content))
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}

	if chunk.ToolCallsGeneration == nil {
		return
	}

	for i, toolCall := range chunk.ToolCallsGeneration.ToolCalls {

		name := toolCall.Name
		id := fmt.Sprintf("tool-%d", i)

		paramBytes, err := json.Marshal(toolCall.GetParameters())
		if err != nil {
			errnie.New(errnie.WithError(err))
			return
		}

		outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
			`{"name": "%s", "id": "%s", "arguments": %s}`,
			name,
			id,
			string(paramBytes),
		), name))

		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}
}

func WithCohereAPIKey(apiKey string) CohereProviderOption {
	return func(prvdr *CohereProvider) {
		prvdr.client = cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		)
	}
}
