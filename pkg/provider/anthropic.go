package provider

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
AnthropicProvider implements an LLM provider that connects to Anthropic's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type AnthropicProvider struct {
	client *anthropic.Client
}

type AnthropicProviderOption func(*AnthropicProvider)

/*
NewAnthropicProvider creates a new Anthropic provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the ANTHROPIC_API_KEY environment variable.
*/
func NewAnthropicProvider(opts ...AnthropicProviderOption) *AnthropicProvider {
	prvdr := &AnthropicProvider{}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *AnthropicProvider) prepare(
	ctx fiber.Ctx, request *task.TaskRequest,
) *anthropic.MessageNewParams {
	params := &anthropic.MessageNewParams{
		Model:       anthropic.Model(tweaker.GetModel(tweaker.GetProvider())),
		Temperature: anthropic.Float(tweaker.GetTemperature()),
		TopP:        anthropic.Float(tweaker.GetTopP()),
	}

	// Convert messages
	msgParams := make([]anthropic.MessageParam, 0)
	var systemMessage string

	for _, msg := range request.Params.History {
		switch msg.Role.String() {
		case "system":
			systemMessage = msg.String()
		case "user":
			msgParams = append(msgParams, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.String())))
		case "assistant":
			msgParams = append(msgParams, anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.String())))
		case "tool":
			msgParams = append(msgParams, anthropic.NewUserMessage(
				anthropic.NewTextBlock(fmt.Sprintf("[Tool Result: %s]", msg.String())),
			))
		}
	}

	// Set system message if present
	if systemMessage != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemMessage},
		}
	}

	params.Messages = msgParams

	// Add tools from registry
	toolParams := make([]anthropic.ToolUnionParam, 0)
	for _, tool := range tools.NewRegistry().GetToolNames() {
		toolParams = append(toolParams, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name: tool,
				InputSchema: anthropic.ToolInputSchemaParam{
					Type: "object",
				},
			},
		})
	}
	params.Tools = toolParams

	return params
}

func (prvdr *AnthropicProvider) Generate(
	ctx fiber.Ctx, request *task.TaskRequest,
) (<-chan *task.TaskResponse, error) {
	out := make(chan *task.TaskResponse)

	go func() {
		defer close(out)

		var (
			params  = prvdr.prepare(ctx, request)
			outTask = request.Params
		)

		response, err := prvdr.client.Messages.New(ctx.Context(), *params)
		if err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		if response.Content == nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(errors.New("content is nil")))
			return
		}

		for _, block := range response.Content {
			switch block := block.AsAny().(type) {
			case anthropic.TextBlock:
				outTask.AddMessage(task.NewAssistantMessage(block.Text))
			case anthropic.ToolUseBlock:
				toolJSON, err := json.Marshal(block.JSON.Input)
				if err != nil {
					errnie.Error("failed to marshal tool input", "error", err)
					continue
				}

				outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
					`{"name": "%s", "arguments": %s}`,
					block.Name,
					string(toolJSON),
				)))
			}
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *AnthropicProvider) Stream(
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

		stream := prvdr.client.Messages.NewStreaming(ctx.Context(), *params)
		defer stream.Close()

		accumulatedMessage := anthropic.Message{}

		for stream.Next() {
			chunk := stream.Current()

			prvdr.handleChunk(chunk, &accumulatedMessage, outTask, out)
		}

		if err := stream.Err(); err != nil {
			outTask.Status.State = task.TaskStateFailed
			out <- task.NewTaskResponse(task.WithResponseError(err))
			return
		}

		outTask.Status.State = task.TaskStateCompleted
		out <- task.NewTaskResponse(task.WithResponseTask(outTask))
	}()

	return out, nil
}

func (prvdr *AnthropicProvider) handleChunk(
	chunk anthropic.MessageStreamEventUnion,
	accumulatedMessage *anthropic.Message,
	outTask task.Task,
	out chan *task.TaskResponse,
) {
	if err := accumulatedMessage.Accumulate(chunk); err != nil {
		outTask.Status.State = task.TaskStateFailed
		out <- task.NewTaskResponse(task.WithResponseError(err))
		return
	}

	switch event := chunk.AsAny().(type) {
	case anthropic.ContentBlockDeltaEvent:
		if event.Delta.Text != "" {
			outTask.AddMessage(task.NewAssistantMessage(event.Delta.Text))
			out <- task.NewTaskResponse(task.WithResponseTask(outTask))
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

				prvdr.handleToolCall(toolData, outTask, out)
			}
		}
	}
}

func (prvdr *AnthropicProvider) handleToolCall(
	toolData []byte,
	outTask task.Task,
	out chan *task.TaskResponse,
) {
	var toolInfo struct {
		ID    string         `json:"id"`
		Name  string         `json:"name"`
		Input map[string]any `json:"input"`
	}

	if err := json.Unmarshal(toolData, &toolInfo); err != nil {
		errnie.Error("failed to unmarshal tool data", "error", err)
		return
	}

	toolJSON, err := json.Marshal(toolInfo.Input)
	if err != nil {
		errnie.Error("failed to marshal tool input", "error", err)
		return
	}

	outTask.AddMessage(task.NewToolMessage(fmt.Sprintf(
		`{"name": "%s", "arguments": %s}`,
		toolInfo.Name,
		string(toolJSON),
	)))

	out <- task.NewTaskResponse(task.WithResponseTask(outTask))

}

func WithAnthropicAPIKey(apiKey string) AnthropicProviderOption {
	return func(provider *AnthropicProvider) {
		client := anthropic.NewClient(
			option.WithAPIKey(apiKey),
		)

		provider.client = &client
	}
}
