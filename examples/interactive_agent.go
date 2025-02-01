package examples

import (
	"context"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

func RunInteractiveAgentExample() {
	ctx := drknow.QuickContext(`
		You are Dr. Know, an AI assistant that can interact with web applications.
		You have access to a browser tool that can navigate websites, click elements,
		fill forms, and extract information.
		
		When performing web interactions, you should:
		1. Plan the sequence of interactions needed
		2. Handle any errors or unexpected states
		3. Verify the results of each action
		4. Report progress and results clearly
	`)

	agent := ai.NewAgent(ctx, provider.NewBalancedProvider(), "interactive", 8)

	msg := provider.NewMessage(
		provider.RoleUser,
		"Go to pkg.go.dev, search for 'web framework', and analyze the top 3 results. Compare their features, popularity, and recent activity.",
	)

	stream.NewConsumer().Print(
		agent.Generate(context.Background(), msg),
		false,
	)
} 