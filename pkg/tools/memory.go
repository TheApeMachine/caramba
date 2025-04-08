package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
)

/* MemoryTool provides a base for all memory operations */
type MemoryTool struct {
	operations map[string]ToolType
}

/* NewMemoryTool creates a new Memory tool with all operations */
func NewMemoryTool(artifact *datura.Artifact) *MemoryTool {
	query := NewMemoryQueryTool(artifact)
	store := NewMemoryStoreTool(artifact)
	search := NewMemorySearchTool(artifact)

	return &MemoryTool{
		operations: map[string]ToolType{
			"memory_query":  {query.Tool, query.Use, query.UseMCP},
			"memory_store":  {store.Tool, store.Use, store.UseMCP},
			"memory_search": {search.Tool, search.Use, search.UseMCP},
		},
	}
}

func (tool *MemoryTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.operations[toolName].Use(ctx, artifact)
}

/* ToMCP returns all Memory tool definitions */
func (tool *MemoryTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* MemoryQueryTool implements a tool for querying memory stores */
type MemoryQueryTool struct {
	mcp.Tool
}

/* NewMemoryQueryTool creates a new tool for memory queries */
func NewMemoryQueryTool(artifact *datura.Artifact) *MemoryQueryTool {
	return &MemoryQueryTool{
		Tool: mcp.NewTool(
			"memory_query",
			mcp.WithDescription("A tool for querying memory stores with specific patterns."),
			mcp.WithString(
				"query",
				mcp.Description("The query pattern to search for."),
				mcp.Required(),
			),
			mcp.WithString(
				"store",
				mcp.Description("The memory store to query (vector/graph)."),
				mcp.Enum("vector", "graph"),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the memory query operation and returns the results */
func (tool *MemoryQueryTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *MemoryQueryTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* MemoryStoreTool implements a tool for storing data in memory */
type MemoryStoreTool struct {
	mcp.Tool
}

/* NewMemoryStoreTool creates a new tool for storing memory */
func NewMemoryStoreTool(artifact *datura.Artifact) *MemoryStoreTool {
	return &MemoryStoreTool{
		Tool: mcp.NewTool(
			"memory_store",
			mcp.WithDescription("A tool for storing data in memory stores."),
			mcp.WithString(
				"data",
				mcp.Description("The data to store."),
				mcp.Required(),
			),
			mcp.WithString(
				"store",
				mcp.Description("The memory store to use (vector/graph)."),
				mcp.Enum("vector", "graph"),
				mcp.Required(),
			),
			mcp.WithString(
				"metadata",
				mcp.Description("Additional metadata to store with the data."),
			),
		),
	}
}

/* Use executes the memory store operation and returns the results */
func (tool *MemoryStoreTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *MemoryStoreTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}

/* MemorySearchTool implements a tool for semantic search in memory */
type MemorySearchTool struct {
	mcp.Tool
}

/* NewMemorySearchTool creates a new tool for memory search */
func NewMemorySearchTool(artifact *datura.Artifact) *MemorySearchTool {
	return &MemorySearchTool{
		Tool: mcp.NewTool(
			"memory_search",
			mcp.WithDescription("A tool for semantic search across memory stores."),
			mcp.WithString(
				"query",
				mcp.Description("The semantic query to search for."),
				mcp.Required(),
			),
			mcp.WithNumber(
				"limit",
				mcp.Description("Maximum number of results to return."),
			),
			mcp.WithNumber(
				"threshold",
				mcp.Description("Similarity threshold for matches."),
			),
		),
	}
}

/* Use executes the memory search operation and returns the results */
func (tool *MemorySearchTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	return artifact
}

func (tool *MemorySearchTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}
