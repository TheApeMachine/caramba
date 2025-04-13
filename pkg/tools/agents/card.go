package agents

import (
	"context"
	"time"

	"github.com/gofiber/fiber/v3/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Card struct {
	mcp.Tool
}

func NewCard() *Card {
	return &Card{
		Tool: mcp.NewTool(
			"agent_card",
			mcp.WithDescription("Get the card of the agent to inspect its details."),
			mcp.WithString(
				"agent_url",
				mcp.Description("The URL of the agent card to get."),
				mcp.Required(),
			),
		),
	}
}

func (card *Card) Use(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	url := req.Params.Arguments["agent_url"].(string)

	if url == "" {
		return mcp.NewToolResultText("agent URL is required"), nil
	}

	resp, err := card.Get(url)
	if err != nil {
		switch resp.StatusCode() {
		case 400:
			return mcp.NewToolResultText("bad request"), nil
		case 401:
			return mcp.NewToolResultText("unauthorized"), nil
		case 404:
			return mcp.NewToolResultText("agent card not found for url: " + url), nil
		default:
			return mcp.NewToolResultText("error getting agent card"), errnie.New(errnie.WithError(err))
		}
	}

	return mcp.NewToolResultText(string(resp.Body())), nil
}

func (card *Card) Get(url string) (*client.Response, error) {
	errnie.Debug("POST", "url", url)

	return client.Post(
		url,
		client.Config{
			Header: map[string]string{
				"Content-Type": "application/json",
			},
			Timeout: 10 * time.Second,
		},
	)
}
