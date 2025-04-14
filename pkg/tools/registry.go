package tools

import (
	"encoding/json"
	"errors"
	"sync"

	"github.com/gofiber/fiber/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/task"
)

var once sync.Once
var registry *Registry

type Registry struct {
	Tools map[string]*Tool
}

func NewRegistry() *Registry {
	once.Do(func() {
		registry = &Registry{}
	})

	return registry
}

func (registry *Registry) Register(tool *Tool) {
	registry.Tools[tool.Tool.Name] = tool
}

func (registry *Registry) GetToolNames() []string {
	names := make([]string, len(registry.Tools))

	for name := range registry.Tools {
		names = append(names, name)
	}

	return names
}

func (registry *Registry) GetTool(name string) *Tool {
	for _, tool := range registry.Tools {
		if tool.Tool.Name == name {
			return tool
		}
	}

	return nil
}

func (registry *Registry) ToOpenAI() []openai.ChatCompletionToolParam {
	toolsOut := make([]openai.ChatCompletionToolParam, 0, len(registry.Tools))

	for _, tool := range registry.Tools {
		mcpTool := tool.Tool

		toolParam := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        mcpTool.Name,
				Description: param.NewOpt(mcpTool.Description),
				Parameters: openai.FunctionParameters{
					"type":       mcpTool.InputSchema.Type,
					"properties": mcpTool.InputSchema.Properties,
				},
			},
		}
		toolsOut = append(toolsOut, toolParam)
	}

	return toolsOut
}

func (registry *Registry) CallOpenAI(
	ctx fiber.Ctx, toolCall openai.ChatCompletionMessageToolCall,
) task.Message {
	tool := registry.GetTool(toolCall.Function.Name)

	if tool == nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(
				errors.New(toolCall.Function.Name+" not found"),
			)).Error(),
			toolCall.Function.Name,
		)
	}

	var args map[string]any

	if err := json.Unmarshal(
		[]byte(toolCall.Function.Arguments), &args,
	); err != nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(err)).Error(),
			toolCall.Function.Name,
		)
	}

	result, err := tool.Use(ctx.Context(), mcp.CallToolRequest{
		Params: struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments,omitempty"`
			Meta      *struct {
				ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
			} `json:"_meta,omitempty"`
		}{
			Name:      toolCall.Function.Name,
			Arguments: args,
			Meta: &struct {
				ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
			}{
				ProgressToken: mcp.ProgressToken(toolCall.ID),
			},
		},
	})

	if err != nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(err)).Error(),
			toolCall.Function.Name,
		)
	}

	var content string

	for _, c := range result.Content {
		switch c := c.(type) {
		case mcp.TextContent:
			content += c.Text
		}
	}

	return task.NewToolMessage(content, toolCall.Function.Name)
}

func (registry *Registry) CallOpenAITool(
	ctx fiber.Ctx,
	toolCall openai.FinishedChatCompletionToolCall,
) task.Message {
	tool := registry.GetTool(toolCall.Name)

	if tool == nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(
				errors.New(toolCall.Name+" not found"),
			)).Error(),
			toolCall.Name,
		)
	}

	var args map[string]any

	if err := json.Unmarshal([]byte(toolCall.Arguments), &args); err != nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(err)).Error(),
			toolCall.Name,
		)
	}

	result, err := tool.Use(ctx.Context(), mcp.CallToolRequest{
		Params: struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments,omitempty"`
			Meta      *struct {
				ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
			} `json:"_meta,omitempty"`
		}{
			Name:      toolCall.Name,
			Arguments: args,
			Meta: &struct {
				ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
			}{
				ProgressToken: mcp.ProgressToken(toolCall.Id),
			},
		},
	})

	if err != nil {
		return task.NewToolMessage(
			errnie.New(errnie.WithError(err)).Error(),
			toolCall.Name,
		)
	}

	var content string

	for _, c := range result.Content {
		switch c := c.(type) {
		case mcp.TextContent:
			content += c.Text
		}
	}

	return task.NewToolMessage(content, toolCall.Name)
}
