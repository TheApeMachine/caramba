package agent

import (
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/stores/neo4j"
	"github.com/theapemachine/caramba/pkg/stores/qdrant"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tools"
)

type Builder struct {
	Card      *Card
	a2aClient *client.A2AClient
	mcpClient *client.MCPClient
	SubAgents []*Builder
	Tasks     []*task.Task
	Provider  provider.ProviderType
	Memory    *memory.Store
	Tools     []tools.Tool
	Inbox     chan *task.TaskRequest
	Outbox    chan *task.TaskResponse
}

type BuilderOption func(*Builder)

func NewBuilder(opts ...BuilderOption) *Builder {
	builder := &Builder{}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func WithCard(card *Card) BuilderOption {
	return func(builder *Builder) {
		builder.Card = card
	}
}

func WithA2AClient(a2aClient *client.A2AClient) BuilderOption {
	return func(builder *Builder) {
		builder.a2aClient = a2aClient
	}
}

func WithMCPClient(mcpClient *client.MCPClient) BuilderOption {
	return func(builder *Builder) {
		builder.mcpClient = mcpClient
	}
}

func WithSubAgents(subAgents []*Builder) BuilderOption {
	return func(builder *Builder) {
		builder.SubAgents = subAgents
	}
}

// WithMemory configures the memory store for the agent with both vector and graph capabilities
func WithMemory(vectorStore *qdrant.Qdrant, graphStore *neo4j.Neo4j, embedder provider.EmbedderType) BuilderOption {
	return func(builder *Builder) {
		builder.Memory = memory.NewStore(
			memory.WithVectorStore(vectorStore),
			memory.WithGraphStore(graphStore),
			memory.WithEmbedder(embedder),
		)
	}
}
