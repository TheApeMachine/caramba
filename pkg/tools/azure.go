package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/tools/azure"
)

/* AzureTool provides a base for all Azure operations */
type AzureTool struct {
	operations map[string]ToolType
}

/* NewAzureTool creates a new Azure tool with all operations */
func NewAzureTool(artifact datura.Artifact) *AzureTool {
	createWorkItem := NewAzureCreateWorkItemTool(artifact)
	updateWorkItem := NewAzureUpdateWorkItemTool(artifact)
	getWorkItem := NewAzureGetWorkItemTool(artifact)
	listWorkItems := NewAzureListWorkItemsTool(artifact)
	createWikiPage := NewAzureCreateWikiPageTool(artifact)
	updateWikiPage := NewAzureUpdateWikiPageTool(artifact)
	getWikiPage := NewAzureGetWikiPageTool(artifact)
	listWikiPages := NewAzureListWikiPagesTool(artifact)

	return &AzureTool{
		operations: map[string]ToolType{
			"create_work_item": {createWorkItem.Tool, createWorkItem.Use, createWorkItem.UseMCP},
			"update_work_item": {updateWorkItem.Tool, updateWorkItem.Use, updateWorkItem.UseMCP},
			"get_work_item":    {getWorkItem.Tool, getWorkItem.Use, getWorkItem.UseMCP},
			"list_work_items":  {listWorkItems.Tool, listWorkItems.Use, listWorkItems.UseMCP},
			"create_wiki_page": {createWikiPage.Tool, createWikiPage.Use, createWikiPage.UseMCP},
			"update_wiki_page": {updateWikiPage.Tool, updateWikiPage.Use, updateWikiPage.UseMCP},
			"get_wiki_page":    {getWikiPage.Tool, getWikiPage.Use, getWikiPage.UseMCP},
			"list_wiki_pages":  {listWikiPages.Tool, listWikiPages.Use, listWikiPages.UseMCP},
		},
	}
}

func (tool *AzureTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.operations[toolName].Use(ctx, artifact)
}

/* ToMCP returns all Azure tool definitions */
func (tool *AzureTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* AzureCreateWorkItemTool implements a tool for creating work items */
type AzureCreateWorkItemTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureCreateWorkItemTool creates a new tool for creating work items */
func NewAzureCreateWorkItemTool(artifact datura.Artifact) *AzureCreateWorkItemTool {
	return &AzureCreateWorkItemTool{
		Tool: mcp.NewTool(
			"create_work_item",
			mcp.WithDescription("A tool for creating work items in Azure DevOps Boards."),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the work item creation operation */
func (tool *AzureCreateWorkItemTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureCreateWorkItemTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureUpdateWorkItemTool implements a tool for updating work items */
type AzureUpdateWorkItemTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureUpdateWorkItemTool creates a new tool for updating work items */
func NewAzureUpdateWorkItemTool(artifact datura.Artifact) *AzureUpdateWorkItemTool {
	return &AzureUpdateWorkItemTool{
		Tool: mcp.NewTool(
			"update_work_item",
			mcp.WithDescription("A tool for updating work items in Azure DevOps Boards."),
			mcp.WithString(
				"work_item_id",
				mcp.Description("The ID of the work item to update."),
				mcp.Required(),
			),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the work item update operation */
func (tool *AzureUpdateWorkItemTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureUpdateWorkItemTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureGetWorkItemTool implements a tool for getting work items */
type AzureGetWorkItemTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureGetWorkItemTool creates a new tool for getting work items */
func NewAzureGetWorkItemTool(artifact datura.Artifact) *AzureGetWorkItemTool {
	return &AzureGetWorkItemTool{
		Tool: mcp.NewTool(
			"get_work_item",
			mcp.WithDescription("A tool for getting work items in Azure DevOps Boards."),
			mcp.WithString(
				"work_item_id",
				mcp.Description("The ID of the work item to get."),
				mcp.Required(),
			),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the work item retrieval operation */
func (tool *AzureGetWorkItemTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureGetWorkItemTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureListWorkItemsTool implements a tool for listing work items */
type AzureListWorkItemsTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureListWorkItemsTool creates a new tool for listing work items */
func NewAzureListWorkItemsTool(artifact datura.Artifact) *AzureListWorkItemsTool {
	return &AzureListWorkItemsTool{
		Tool: mcp.NewTool(
			"list_work_items",
			mcp.WithDescription("A tool for listing work items in Azure DevOps Boards."),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the work item listing operation */
func (tool *AzureListWorkItemsTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureListWorkItemsTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureCreateWikiPageTool implements a tool for creating wiki pages */
type AzureCreateWikiPageTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureCreateWikiPageTool creates a new tool for creating wiki pages */
func NewAzureCreateWikiPageTool(artifact datura.Artifact) *AzureCreateWikiPageTool {
	return &AzureCreateWikiPageTool{
		Tool: mcp.NewTool(
			"create_wiki_page",
			mcp.WithDescription("A tool for creating wiki pages in Azure DevOps."),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the wiki page creation operation */
func (tool *AzureCreateWikiPageTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureCreateWikiPageTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureUpdateWikiPageTool implements a tool for updating wiki pages */
type AzureUpdateWikiPageTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureUpdateWikiPageTool creates a new tool for updating wiki pages */
func NewAzureUpdateWikiPageTool(artifact datura.Artifact) *AzureUpdateWikiPageTool {
	return &AzureUpdateWikiPageTool{
		Tool: mcp.NewTool(
			"update_wiki_page",
			mcp.WithDescription("A tool for updating wiki pages in Azure DevOps."),
			mcp.WithString(
				"wiki_page_id",
				mcp.Description("The ID of the wiki page to update."),
				mcp.Required(),
			),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the wiki page update operation */
func (tool *AzureUpdateWikiPageTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureUpdateWikiPageTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureGetWikiPageTool implements a tool for getting wiki pages */
type AzureGetWikiPageTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureGetWikiPageTool creates a new tool for getting wiki pages */
func NewAzureGetWikiPageTool(artifact datura.Artifact) *AzureGetWikiPageTool {
	return &AzureGetWikiPageTool{
		Tool: mcp.NewTool(
			"get_wiki_page",
			mcp.WithDescription("A tool for getting wiki pages in Azure DevOps."),
			mcp.WithString(
				"wiki_page_id",
				mcp.Description("The ID of the wiki page to get."),
				mcp.Required(),
			),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the wiki page retrieval operation */
func (tool *AzureGetWikiPageTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureGetWikiPageTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* AzureListWikiPagesTool implements a tool for listing wiki pages */
type AzureListWikiPagesTool struct {
	mcp.Tool
	client *azure.Client
}

/* NewAzureListWikiPagesTool creates a new tool for listing wiki pages */
func NewAzureListWikiPagesTool(artifact datura.Artifact) *AzureListWikiPagesTool {
	return &AzureListWikiPagesTool{
		Tool: mcp.NewTool(
			"list_wiki_pages",
			mcp.WithDescription("A tool for listing wiki pages in Azure DevOps."),
		),
		client: azure.NewClient(),
	}
}

/* Use executes the wiki page listing operation */
func (tool *AzureListWikiPagesTool) Use(
	ctx context.Context, artifact datura.Artifact,
) datura.Artifact {
	return artifact
}

func (tool *AzureListWikiPagesTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}
