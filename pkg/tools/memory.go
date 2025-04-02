package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
)

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
It handles both vector-based (Qdrant) and graph-based (Neo4j) storage systems.
*/
type MemoryTool struct {
	ctx    context.Context
	cancel context.CancelFunc
	stores map[string]interface{} // Map of store types to their implementations
	*ToolBuilder
}

type MemoryToolOption func(*MemoryTool)

// NewMemoryTool creates a new memory tool with the specified stores.
// If no stores are provided, it initializes with default Qdrant and Neo4j stores.
func NewMemoryTool(opts ...MemoryToolOption) *MemoryTool {
	errnie.Debug("NewMemoryTool")

	ctx, cancel := context.WithCancel(context.Background())

	// Create MCP tool definition for memory tool
	memoryTool := mcp.NewTool(
		"memory",
		mcp.WithDescription("A tool which can interact with various memory stores through a unified interface."),
	)

	builder := NewToolBuilder()
	builder.mcp = &memoryTool

	tool := &MemoryTool{
		ctx:         ctx,
		cancel:      cancel,
		stores:      make(map[string]interface{}),
		ToolBuilder: builder,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func (mt *MemoryTool) ID() string {
	return "memory"
}

func (mt *MemoryTool) ToMCP() mcp.Tool {
	return *mt.ToolBuilder.mcp
}

// Generate implements the Generator pattern for MemoryTool.
// It processes queries and returns results through the artifact channel.
func (mt *MemoryTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("tools.MemoryTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

	}()

	return out
}

func WithStores(stores ...memory.Store) MemoryToolOption {
	return func(tool *MemoryTool) {
		for _, store := range stores {
			tool.stores[store.Name()] = store
		}
	}
}

func WithStore(store memory.Store) MemoryToolOption {
	return func(tool *MemoryTool) {
		tool.stores[store.Name()] = store
	}
}
