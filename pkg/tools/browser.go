package tools

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/browser"
)

func init() {
	fmt.Println("tools.browser.init")
	provider.RegisterTool("browser")
}

// BrowserBaseTool provides common functionality for all browser tools
type BrowserBaseTool struct {
	pctx      context.Context
	ctx       context.Context
	cancel    context.CancelFunc
	instance  *browser.Manager
	Schema    *provider.Tool
	operation string
}

// Generate handles the common generation logic for all browser tools
func (bbt *BrowserBaseTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("browser.BrowserBaseTool.Generate." + bbt.operation)

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-bbt.pctx.Done():
			errnie.Debug("browser.BrowserBaseTool.Generate.pctx.Done." + bbt.operation)
			bbt.cancel()
			return
		case <-bbt.ctx.Done():
			errnie.Debug("browser.BrowserBaseTool.Generate.ctx.Done." + bbt.operation)
			bbt.cancel()
			return
		case artifact := <-buffer:
			artifact.SetMetaValue("operation", bbt.operation)

			manager, err := browser.NewManager(artifact).Initialize()
			if err != nil {
				errnie.Error(err)
				artifact.SetMetaValue("error", err.Error())
				out <- artifact
				return
			}
			defer manager.Close()

			switch bbt.operation {
			case "get_content":
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
			case "get_links":
				// Use eval for link extraction
				val, err := browser.NewEval(manager.GetPage(), artifact, "get_links").Run()
				if err != nil {
					errnie.Error(err)
					artifact.SetMetaValue("error", err.Error())
				} else {
					datura.WithPayload([]byte(val))(artifact)
				}
			default:
				val, err := browser.NewEval(manager.GetPage(), artifact, bbt.operation).Run()
				if err != nil {
					errnie.Error(err)
					artifact.SetMetaValue("error", err.Error())
				} else {
					datura.WithPayload([]byte(val))(artifact)
				}
			}

			out <- artifact
		}
	}()

	return out
}

// NewBrowserBaseTool creates a new base browser tool with the specified schema and operation
func NewBrowserBaseTool(operation string) *BrowserBaseTool {
	instance := browser.NewManager(datura.New())

	return &BrowserBaseTool{
		instance:  instance,
		Schema:    GetToolSchema("browser"),
		operation: operation,
	}
}

// WithBrowserCancelBase sets the parent context for a browser base tool
func WithBrowserCancelBase(ctx context.Context) func(*BrowserBaseTool) {
	return func(tool *BrowserBaseTool) {
		tool.pctx = ctx
	}
}

// BrowserGetContentTool implements a tool for retrieving page content
type BrowserGetContentTool struct {
	*BrowserBaseTool
}

// NewBrowserGetContentTool creates a new tool for retrieving page content
func NewBrowserGetContentTool() *BrowserGetContentTool {
	return &BrowserGetContentTool{
		BrowserBaseTool: NewBrowserBaseTool("get_content"),
	}
}

// Generate forwards the generation to the base tool
func (bgct *BrowserGetContentTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return bgct.BrowserBaseTool.Generate(buffer)
}

// WithCancelBrowserGetContentTool sets the parent context for a browser get content tool
func WithCancelBrowserGetContentTool(ctx context.Context) func(*BrowserGetContentTool) {
	return func(tool *BrowserGetContentTool) {
		WithBrowserCancelBase(ctx)(tool.BrowserBaseTool)
	}
}

// BrowserGetLinksTool implements a tool for extracting links from a page
type BrowserGetLinksTool struct {
	*BrowserBaseTool
}

// NewBrowserGetLinksTool creates a new tool for extracting links from a page
func NewBrowserGetLinksTool() *BrowserGetLinksTool {
	return &BrowserGetLinksTool{
		BrowserBaseTool: NewBrowserBaseTool("get_links"),
	}
}

// Generate forwards the generation to the base tool
func (bglt *BrowserGetLinksTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return bglt.BrowserBaseTool.Generate(buffer)
}

// WithCancelBrowserGetLinksTool sets the parent context for a browser get links tool
func WithCancelBrowserGetLinksTool(ctx context.Context) func(*BrowserGetLinksTool) {
	return func(tool *BrowserGetLinksTool) {
		WithBrowserCancelBase(ctx)(tool.BrowserBaseTool)
	}
}
