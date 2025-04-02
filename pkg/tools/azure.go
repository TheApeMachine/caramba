package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/azure"
)

// AzureTool provides common functionality for all Azure tools
type AzureTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	client *azure.Client
}

type AzureToolOption func(*AzureTool)

// NewAzureTool creates a new Azure tool with the specified options
func NewAzureTool(opts ...AzureToolOption) *AzureTool {
	ctx, cancel := context.WithCancel(context.Background())

	client := azure.NewClient()

	tool := &AzureTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
		client:      client,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

// WithAzureCancel sets the parent context for an Azure tool
func WithAzureCancel(ctx context.Context) AzureToolOption {
	return func(tool *AzureTool) {
		tool.pctx = ctx
	}
}

// Generate handles the common generation logic for all Azure tools
func (tool *AzureTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("azure.AzureTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("azure.AzureTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("azure.AzureTool.Generate: context done")
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

// AzureCreateWorkItemTool implements a tool for creating work items
type AzureCreateWorkItemTool struct {
	*AzureTool
}

// NewAzureCreateWorkItemTool creates a new tool for creating work items
func NewAzureCreateWorkItemTool() *AzureCreateWorkItemTool {
	// Create MCP tool definition based on schema from config.yml
	createWorkItemTool := mcp.NewTool(
		"create_work_item",
		mcp.WithDescription("A tool for creating work items in Azure DevOps Boards."),
	)

	acwit := &AzureCreateWorkItemTool{
		AzureTool: NewAzureTool(),
	}

	acwit.ToolBuilder.mcp = &createWorkItemTool
	return acwit
}

func (tool *AzureCreateWorkItemTool) ID() string {
	return "azure_create_work_item"
}

// Generate processes the work item creation operation
func (tool *AzureCreateWorkItemTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the work item creation operation
func (tool *AzureCreateWorkItemTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureCreateWorkItemTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "create_work_item")

	// Implementation for creating work items
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureCreateWorkItemTool
func (tool *AzureCreateWorkItemTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureUpdateWorkItemTool implements a tool for updating work items
type AzureUpdateWorkItemTool struct {
	*AzureTool
}

// NewAzureUpdateWorkItemTool creates a new tool for updating work items
func NewAzureUpdateWorkItemTool() *AzureUpdateWorkItemTool {
	// Create MCP tool definition based on schema from config.yml
	updateWorkItemTool := mcp.NewTool(
		"update_work_item",
		mcp.WithDescription("A tool for interacting with Azure DevOps Boards and Wikis."),
		mcp.WithString(
			"work_item_id",
			mcp.Description("The ID of the work item to update."),
			mcp.Required(),
		),
	)

	auwit := &AzureUpdateWorkItemTool{
		AzureTool: NewAzureTool(),
	}

	auwit.ToolBuilder.mcp = &updateWorkItemTool
	return auwit
}

func (tool *AzureUpdateWorkItemTool) ID() string {
	return "azure_update_work_item"
}

// Generate processes the work item update operation
func (tool *AzureUpdateWorkItemTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the work item update operation
func (tool *AzureUpdateWorkItemTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureUpdateWorkItemTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "update_work_item")

	// Implementation for updating work items
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureUpdateWorkItemTool
func (tool *AzureUpdateWorkItemTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureGetWorkItemTool implements a tool for getting work items
type AzureGetWorkItemTool struct {
	*AzureTool
}

// NewAzureGetWorkItemTool creates a new tool for getting work items
func NewAzureGetWorkItemTool() *AzureGetWorkItemTool {
	// Create MCP tool definition based on schema from config.yml
	getWorkItemTool := mcp.NewTool(
		"get_work_item",
		mcp.WithDescription("A tool for getting a work item in Azure DevOps Boards."),
		mcp.WithString(
			"work_item_id",
			mcp.Description("The ID of the work item to get."),
			mcp.Required(),
		),
	)

	agwit := &AzureGetWorkItemTool{
		AzureTool: NewAzureTool(),
	}

	agwit.ToolBuilder.mcp = &getWorkItemTool
	return agwit
}

func (tool *AzureGetWorkItemTool) ID() string {
	return "azure_get_work_item"
}

// Generate processes the work item retrieval operation
func (tool *AzureGetWorkItemTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the work item retrieval operation
func (tool *AzureGetWorkItemTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureGetWorkItemTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "get_work_item")

	// Implementation for getting work items
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureGetWorkItemTool
func (tool *AzureGetWorkItemTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureListWorkItemsTool implements a tool for listing work items
type AzureListWorkItemsTool struct {
	*AzureTool
}

// NewAzureListWorkItemsTool creates a new tool for listing work items
func NewAzureListWorkItemsTool() *AzureListWorkItemsTool {
	// Create MCP tool definition based on schema from config.yml
	listWorkItemsTool := mcp.NewTool(
		"list_work_items",
		mcp.WithDescription("A tool for listing work items in Azure DevOps Boards."),
	)

	alwit := &AzureListWorkItemsTool{
		AzureTool: NewAzureTool(),
	}

	alwit.ToolBuilder.mcp = &listWorkItemsTool
	return alwit
}

func (tool *AzureListWorkItemsTool) ID() string {
	return "azure_list_work_items"
}

// Generate processes the work item listing operation
func (tool *AzureListWorkItemsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the work item listing operation
func (tool *AzureListWorkItemsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureListWorkItemsTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "list_work_items")

	// Implementation for listing work items
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureListWorkItemsTool
func (tool *AzureListWorkItemsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureCreateWikiPageTool implements a tool for creating wiki pages
type AzureCreateWikiPageTool struct {
	*AzureTool
}

// NewAzureCreateWikiPageTool creates a new tool for creating wiki pages
func NewAzureCreateWikiPageTool() *AzureCreateWikiPageTool {
	// Create MCP tool definition based on schema from config.yml
	createWikiPageTool := mcp.NewTool(
		"create_wiki_page",
		mcp.WithDescription("A tool for creating wiki pages in Azure DevOps Boards."),
	)

	acwpt := &AzureCreateWikiPageTool{
		AzureTool: NewAzureTool(),
	}

	acwpt.ToolBuilder.mcp = &createWikiPageTool
	return acwpt
}

func (tool *AzureCreateWikiPageTool) ID() string {
	return "azure_create_wiki_page"
}

// Generate processes the wiki page creation operation
func (tool *AzureCreateWikiPageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the wiki page creation operation
func (tool *AzureCreateWikiPageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureCreateWikiPageTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "create_wiki_page")

	// Implementation for creating wiki pages
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureCreateWikiPageTool
func (tool *AzureCreateWikiPageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureUpdateWikiPageTool implements a tool for updating wiki pages
type AzureUpdateWikiPageTool struct {
	*AzureTool
}

// NewAzureUpdateWikiPageTool creates a new tool for updating wiki pages
func NewAzureUpdateWikiPageTool() *AzureUpdateWikiPageTool {
	// Create MCP tool definition based on schema from config.yml
	updateWikiPageTool := mcp.NewTool(
		"update_wiki_page",
		mcp.WithDescription("A tool for updating wiki pages in Azure DevOps Boards."),
		mcp.WithString(
			"wiki_page_id",
			mcp.Description("The ID of the wiki page to update."),
			mcp.Required(),
		),
	)

	auwpt := &AzureUpdateWikiPageTool{
		AzureTool: NewAzureTool(),
	}

	auwpt.ToolBuilder.mcp = &updateWikiPageTool
	return auwpt
}

func (tool *AzureUpdateWikiPageTool) ID() string {
	return "azure_update_wiki_page"
}

// Generate processes the wiki page update operation
func (tool *AzureUpdateWikiPageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the wiki page update operation
func (tool *AzureUpdateWikiPageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureUpdateWikiPageTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "update_wiki_page")

	// Implementation for updating wiki pages
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureUpdateWikiPageTool
func (tool *AzureUpdateWikiPageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureGetWikiPageTool implements a tool for getting wiki pages
type AzureGetWikiPageTool struct {
	*AzureTool
}

// NewAzureGetWikiPageTool creates a new tool for getting wiki pages
func NewAzureGetWikiPageTool() *AzureGetWikiPageTool {
	// Create MCP tool definition based on schema from config.yml
	getWikiPageTool := mcp.NewTool(
		"get_wiki_page",
		mcp.WithDescription("A tool for getting a wiki page in Azure DevOps Boards."),
		mcp.WithString(
			"wiki_page_id",
			mcp.Description("The ID of the wiki page to get."),
			mcp.Required(),
		),
	)

	agwpt := &AzureGetWikiPageTool{
		AzureTool: NewAzureTool(),
	}

	agwpt.ToolBuilder.mcp = &getWikiPageTool
	return agwpt
}

func (tool *AzureGetWikiPageTool) ID() string {
	return "azure_get_wiki_page"
}

// Generate processes the wiki page retrieval operation
func (tool *AzureGetWikiPageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the wiki page retrieval operation
func (tool *AzureGetWikiPageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureGetWikiPageTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "get_wiki_page")

	// Implementation for getting wiki pages
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureGetWikiPageTool
func (tool *AzureGetWikiPageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// AzureListWikiPagesTool implements a tool for listing wiki pages
type AzureListWikiPagesTool struct {
	*AzureTool
}

// NewAzureListWikiPagesTool creates a new tool for listing wiki pages
func NewAzureListWikiPagesTool() *AzureListWikiPagesTool {
	// Create MCP tool definition based on schema from config.yml
	listWikiPagesTool := mcp.NewTool(
		"list_wiki_pages",
		mcp.WithDescription("A tool for listing wiki pages in Azure DevOps Boards."),
	)

	alwpt := &AzureListWikiPagesTool{
		AzureTool: NewAzureTool(),
	}

	alwpt.ToolBuilder.mcp = &listWikiPagesTool
	return alwpt
}

func (tool *AzureListWikiPagesTool) ID() string {
	return "azure_list_wiki_pages"
}

// Generate processes the wiki page listing operation
func (tool *AzureListWikiPagesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.AzureTool.Generate(buffer, tool.fn)
}

// fn implements the wiki page listing operation
func (tool *AzureListWikiPagesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("azure.AzureListWikiPagesTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "list_wiki_pages")

	// Implementation for listing wiki pages
	return artifact
}

// ToMCP returns the MCP tool definitions for the AzureListWikiPagesTool
func (tool *AzureListWikiPagesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
