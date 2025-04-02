package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/browser"
)

// BrowserTool provides common functionality for all browser tools
type BrowserTool struct {
	*ToolBuilder
	pctx     context.Context
	ctx      context.Context
	cancel   context.CancelFunc
	instance *browser.Manager
}

type BrowserToolOption func(*BrowserTool)

// NewBrowserTool creates a new browser tool with the specified options
func NewBrowserTool(opts ...BrowserToolOption) *BrowserTool {
	ctx, cancel := context.WithCancel(context.Background())

	instance := browser.NewManager(datura.New())

	tool := &BrowserTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
		instance:    instance,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

// WithBrowserCancel sets the parent context for a browser tool
func WithBrowserCancel(ctx context.Context) BrowserToolOption {
	return func(tool *BrowserTool) {
		tool.pctx = ctx
	}
}

// Generate handles the common generation logic for all browser tools
func (tool *BrowserTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("browser.BrowserTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("browser.BrowserTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("browser.BrowserTool.Generate: context done")
				return
			case artifact := <-buffer:
				for _, f := range fn {
					out <- f(artifact)
				}
			}
		}
	}()

	return out
}

// BrowserGetContentTool implements a tool for retrieving page content
type BrowserGetContentTool struct {
	*BrowserTool
}

// NewBrowserGetContentTool creates a new tool for retrieving page content
func NewBrowserGetContentTool() *BrowserGetContentTool {
	// Create MCP tool definition based on schema from config.yml
	getContentTool := mcp.NewTool(
		"get_content",
		mcp.WithDescription("A tool which can get the content of a page."),
		mcp.WithString(
			"url",
			mcp.Description("The URL to navigate to."),
			mcp.Required(),
		),
	)

	bgct := &BrowserGetContentTool{
		BrowserTool: NewBrowserTool(),
	}

	bgct.ToolBuilder.mcp = &getContentTool
	return bgct
}

func (tool *BrowserGetContentTool) ID() string {
	return "browser_get_content"
}

// Generate processes the content retrieval operation
func (tool *BrowserGetContentTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.BrowserTool.Generate(buffer, tool.fn)
}

// fn implements the content retrieval operation
func (tool *BrowserGetContentTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("browser.BrowserGetContentTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "get_content")

	manager, err := browser.NewManager(artifact).Initialize()
	if err != nil {
		errnie.Error(err)
		artifact.SetMetaValue("error", err.Error())
		return artifact
	}
	defer manager.Close()

	content, err := manager.GetPage().HTML()
	if err != nil {
		errnie.Error(err)
		artifact.SetMetaValue("error", err.Error())
	} else {
		// Convert HTML to markdown
		markdown, err := browser.ConvertToMarkdown(content)
		if err != nil {
			errnie.Error(err)
			artifact.SetMetaValue("error", err.Error())
		} else {
			datura.WithPayload([]byte(markdown))(artifact)
		}
	}

	return artifact
}

// ToMCP returns the MCP tool definitions for the BrowserGetContentTool
func (tool *BrowserGetContentTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// BrowserGetLinksTool implements a tool for extracting links from a page
type BrowserGetLinksTool struct {
	*BrowserTool
}

// NewBrowserGetLinksTool creates a new tool for extracting links from a page
func NewBrowserGetLinksTool() *BrowserGetLinksTool {
	// Create MCP tool definition based on schema from config.yml
	getLinksTool := mcp.NewTool(
		"get_links",
		mcp.WithDescription("A tool which can get the links of a page."),
		mcp.WithString(
			"url",
			mcp.Description("The URL to navigate to."),
			mcp.Required(),
		),
	)

	bglt := &BrowserGetLinksTool{
		BrowserTool: NewBrowserTool(),
	}

	bglt.ToolBuilder.mcp = &getLinksTool
	return bglt
}

func (tool *BrowserGetLinksTool) ID() string {
	return "browser_get_links"
}

// Generate processes the link extraction operation
func (tool *BrowserGetLinksTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.BrowserTool.Generate(buffer, tool.fn)
}

// fn implements the link extraction operation
func (tool *BrowserGetLinksTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("browser.BrowserGetLinksTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "get_links")

	manager, err := browser.NewManager(artifact).Initialize()
	if err != nil {
		errnie.Error(err)
		artifact.SetMetaValue("error", err.Error())
		return artifact
	}
	defer manager.Close()

	// Use eval for link extraction
	val, err := browser.NewEval(manager.GetPage(), artifact, "get_links").Run()
	if err != nil {
		errnie.Error(err)
		artifact.SetMetaValue("error", err.Error())
	} else {
		datura.WithPayload([]byte(val))(artifact)
	}

	return artifact
}

// ToMCP returns the MCP tool definitions for the BrowserGetLinksTool
func (tool *BrowserGetLinksTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
