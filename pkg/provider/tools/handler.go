package tools

import (
	"github.com/gofiber/fiber/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/registry"
	"github.com/theapemachine/caramba/pkg/tools/interfaces"
)

type ToolCallHandler struct {
	toolCall *mcp.CallToolRequest
}

func NewToolCallHandler(
	toolCall *mcp.CallToolRequest,
) *ToolCallHandler {
	return &ToolCallHandler{
		toolCall: toolCall,
	}
}

func (handler *ToolCallHandler) Handle(ctx fiber.Ctx) (*mcp.CallToolResult, error) {
	reg := registry.GetAmbient()
	rawTool, err := reg.GetTool(handler.toolCall.Method)
	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	tool, ok := rawTool.(interfaces.Tool)
	if !ok {
		return nil, errnie.New(errnie.WithMessage("invalid tool type"))
	}

	return tool.Use(ctx.Context(), *handler.toolCall)
}
