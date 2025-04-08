package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/tools/browser"
)

/* BrowserTool provides a base for all browser operations */
type BrowserTool struct {
	operations map[string]ToolType
}

/* NewBrowserTool creates a new browser tool with all operations */
func NewBrowserTool(artifact *datura.Artifact) *BrowserTool {
	getContent := NewBrowserGetContentTool(artifact)
	getLinks := NewBrowserGetLinksTool(artifact)

	return &BrowserTool{
		operations: map[string]ToolType{
			"get_content": {getContent.Tool, getContent.Use, getContent.UseMCP},
			"get_links":   {getLinks.Tool, getLinks.Use, getLinks.UseMCP},
		},
	}
}

func (tool *BrowserTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.operations[toolName].Use(ctx, artifact)
}

/* ToMCP returns all browser tool definitions */
func (tool *BrowserTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* BrowserGetContentTool implements a tool for retrieving page content */
type BrowserGetContentTool struct {
	mcp.Tool
	instance *browser.Manager
}

/* NewBrowserGetContentTool creates a new tool for retrieving page content */
func NewBrowserGetContentTool(artifact *datura.Artifact) *BrowserGetContentTool {
	return &BrowserGetContentTool{
		Tool: mcp.NewTool(
			"get_content",
			mcp.WithDescription("A tool which can get the content of a page."),
			mcp.WithString(
				"url",
				mcp.Description("The URL to navigate to."),
				mcp.Required(),
			),
		),
		instance: browser.NewManager(artifact),
	}
}

/* Use executes the content retrieval operation */
func (tool *BrowserGetContentTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	instance, err := tool.instance.Initialize()
	if err != nil {
		return artifact
	}
	defer instance.Close()

	content, err := instance.GetPage().HTML()
	if err != nil {
		return artifact
	}

	// Convert HTML to markdown
	markdown, err := browser.ConvertToMarkdown(content)
	if err != nil {
		return artifact
	}

	datura.WithPayload([]byte(markdown))

	return artifact
}

func (tool *BrowserGetContentTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* BrowserGetLinksTool implements a tool for extracting links from a page */
type BrowserGetLinksTool struct {
	mcp.Tool
	instance *browser.Manager
}

/* NewBrowserGetLinksTool creates a new tool for extracting links */
func NewBrowserGetLinksTool(artifact *datura.Artifact) *BrowserGetLinksTool {
	return &BrowserGetLinksTool{
		Tool: mcp.NewTool(
			"get_links",
			mcp.WithDescription("A tool which can get the links of a page."),
			mcp.WithString(
				"url",
				mcp.Description("The URL to navigate to."),
				mcp.Required(),
			),
		),
		instance: browser.NewManager(artifact),
	}
}

/* Use executes the link extraction operation */
func (tool *BrowserGetLinksTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	instance, err := tool.instance.Initialize()
	if err != nil {
		return artifact
	}
	defer instance.Close()

	// Use eval for link extraction
	val, err := browser.NewEval(instance.GetPage(), artifact, "get_links").Run()
	if err != nil {
		return artifact
	}

	datura.WithPayload([]byte(val))

	return artifact
}

func (tool *BrowserGetLinksTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}
