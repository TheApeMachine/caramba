package examples

import (
	"context"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

func RunResearchAgentExample() {
	ctx := drknow.QuickContext(`
		You are Dr. Know, an AI research assistant that helps users gather and analyze information from multiple sources.
		You have access to a browser tool that can navigate websites and extract information.
		
		When conducting research, you should:
		1. Break down the research question into key aspects
		2. Identify multiple authoritative sources
		3. Use the browser tool to gather information from each source
		4. Cross-reference and validate the information
		5. Synthesize findings into a comprehensive response
	`)

	agent := ai.NewAgent(ctx, provider.NewBalancedProvider(), "researcher", 10)

	msg := provider.NewMessage(
		provider.RoleUser,
		"Research the latest developments in Go's generics implementation. Compare information from the Go blog, GitHub discussions, and recent conference talks.",
	)

	stream.NewConsumer().Print(
		agent.Generate(context.Background(), msg),
		false,
	)
} 