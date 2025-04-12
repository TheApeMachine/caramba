package agent

import "github.com/theapemachine/caramba/pkg/service/client"

type Builder struct {
	Card      *Card
	a2aClient *client.A2AClient
	mcpClient *client.MCPClient
}

type BuilderOption func(*Builder)

func NewBuilder(opts ...BuilderOption) *Builder {
	builder := &Builder{
		a2aClient: client.NewA2AClient(
			"http://localhost:3000",
		),
		mcpClient: client.NewMCPClient(),
	}

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
