package agent

import (
	"context"
	"io"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/registry"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/stores/neo4j"
	"github.com/theapemachine/caramba/pkg/stores/qdrant"
	"github.com/theapemachine/caramba/pkg/task"
)

type Agent interface {
	HandleTask(ctx fiber.Ctx, req *task.TaskRequest) error
	AddWriter(w io.Writer)
	Card() *Card
}

type Builder struct {
	ID          string
	card        *Card
	a2aClient   *client.RPCClient
	mcpClient   *client.MCPClient
	Tasks       []*task.Task
	Provider    provider.ProviderType
	Memory      *memory.Store
	taskManager *Manager
	writers     io.Writer
}

type BuilderOption func(*Builder)

func NewBuilder(opts ...BuilderOption) *Builder {
	builder := &Builder{
		ID: uuid.New().String(),
	}

	if err := registry.GetAmbient().Register(
		context.Background(),
		"local_agent",
		builder.ID,
	); err != nil {
		errnie.New(errnie.WithError(err))
		return nil
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func (builder *Builder) AddWriter(w io.Writer) {
	if builder.writers == nil {
		builder.writers = w
	} else {
		builder.writers = io.MultiWriter(builder.writers, w)
	}
}

func (builder *Builder) Card() *Card {
	return builder.card
}

func (builder *Builder) HandleTask(
	ctx fiber.Ctx, req *task.TaskRequest,
) error {
	return builder.taskManager.HandleTask(ctx, req, builder.writers)
}

func WithCard(card *Card) BuilderOption {
	return func(builder *Builder) {
		builder.card = card
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

func WithMemory(vectorStore *qdrant.Qdrant, graphStore *neo4j.Neo4j, embedder provider.EmbedderType) BuilderOption {
	return func(builder *Builder) {
		builder.Memory = memory.NewStore(
			memory.WithVectorStore(vectorStore),
			memory.WithGraphStore(graphStore),
			memory.WithEmbedder(embedder),
		)
	}
}

func WithTaskManager(taskManager *Manager) BuilderOption {
	return func(builder *Builder) {
		builder.taskManager = taskManager
	}
}
