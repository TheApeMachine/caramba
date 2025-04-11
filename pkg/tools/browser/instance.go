package browser

import (
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	fs "github.com/theapemachine/caramba/pkg/stores/fs"
	"github.com/theapemachine/caramba/pkg/utils"
)

// BrowserGenerator implements the stream.Generator interface for browser operations
type BrowserGenerator struct {
	fsStore *fs.Store
}

// Generate processes browser operations
func (bg *BrowserGenerator) Do(toolcall mcp.CallToolRequest) mcp.CallToolResult {
	errnie.Debug("browser.Instance.buffer.fn")

	manager, err := NewManager(toolcall).Initialize()

	if errnie.Error(err) != nil {
		return mcp.CallToolResult{
			Content: []mcp.Content{
				mcp.TextContent{
					Type: "text",
					Text: errnie.Error(err).Error(),
				},
			},
		}
	}

	defer manager.Close()

	op := toolcall.Params.Arguments["operation"].(string)

	switch op {
	case "get_content":
		var (
			content  string
			markdown string
		)

		if content, err = manager.GetPage().HTML(); errnie.Error(err) != nil {
			return mcp.CallToolResult{
				Content: []mcp.Content{
					mcp.TextContent{
						Type: "text",
						Text: errnie.Error(err).Error(),
					},
				},
			}
		}

		if markdown, err = htmltomarkdown.ConvertString(content); errnie.Error(err) != nil {
			return mcp.CallToolResult{
				Content: []mcp.Content{
					mcp.TextContent{
						Type: "text",
						Text: errnie.Error(err).Error(),
					},
				},
			}
		}

		return mcp.CallToolResult{
			Content: []mcp.Content{
				mcp.TextContent{
					Type: "text",
					Text: markdown,
				},
			},
		}
	default:
		var val string

		pendingCall := provider.PendingToolCall{
			ID:      "",
			Request: toolcall,
		}

		if val, err = NewEval(manager.GetPage(), bg.fsStore, pendingCall).Run(); errnie.Error(err) != nil {
			return mcp.CallToolResult{
				Content: []mcp.Content{
					mcp.TextContent{
						Type: "text",
						Text: errnie.Error(err).Error(),
					},
				},
			}
		}

		return mcp.CallToolResult{
			Content: []mcp.Content{
				mcp.TextContent{
					Type: "text",
					Text: utils.SummarizeText(val, 2000),
				},
			},
		}
	}
}
