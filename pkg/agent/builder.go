package agent

import "github.com/theapemachine/caramba/pkg/service/client"

type Builder struct {
	Card      *Card
	a2aClient *client.A2AClient
	mcpClient *client.MCPClient
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
