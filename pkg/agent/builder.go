package agent

import (
	"context"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/registry"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/stores/neo4j"
	"github.com/theapemachine/caramba/pkg/stores/qdrant"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/task/manager"
)

type Builder struct {
	ID          string
	Card        *Card
	a2aClient   *client.A2AClient
	mcpClient   *client.MCPClient
	Tasks       []*task.Task
	Provider    provider.ProviderType
	Memory      *memory.Store
	taskManager *manager.Manager
}

type BuilderOption func(*Builder)

func NewBuilder(opts ...BuilderOption) *Builder {
	builder := &Builder{
		ID: uuid.New().String(),
	}

	registry.GetAmbient().Register(
		context.Background(),
		"local_agent",
		builder.ID,
	)

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func (builder *Builder) HandleTask(
	ctx fiber.Ctx, req *task.TaskRequest,
) error {
	return builder.taskManager.HandleTask(ctx, req)
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
		for _, subAgent := range subAgents {
			registry.GetAmbient().Register(
				context.Background(),
				builder.ID,
				subAgent.Card,
			)
		}
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

func WithTaskManager(taskManager *manager.Manager) BuilderOption {
	return func(builder *Builder) {
		builder.taskManager = taskManager
	}
}
