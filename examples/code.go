package examples

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/theapemachine/caramba/pkg/ai/agent"
	"github.com/theapemachine/caramba/pkg/ai/provider"
	"github.com/theapemachine/caramba/pkg/datura"
	prvdr "github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/twoface"
)

// Code represents a test setup for the agent framework
type Code struct {
	ctx      context.Context
	cancel   context.CancelFunc
	hub      *twoface.Hub
	planner  *agent.AgentBuilder
	dev      *agent.AgentBuilder
	provider *provider.ProviderBuilder
}

// NewCode creates a new test setup for the agent framework
func NewCode() *Code {
	ctx, cancel := context.WithCancel(context.Background())

	return &Code{
		ctx:    ctx,
		cancel: cancel,
		hub:    twoface.NewHub(),
	}
}

// Run executes the test setup
func (code *Code) Run() {
	fmt.Println("Starting agent framework test...")

	// Create agents
	code.planner = agent.New(
		agent.WithName("planner"),
		agent.WithRole("planner"),
		agent.WithModel(tweaker.GetModel("openai")),
		agent.WithTransport(code.hub.NewTransport()),
	)

	code.dev = agent.New(
		agent.WithName("developer"),
		agent.WithRole("developer"),
		agent.WithModel(tweaker.GetModel("openai")),
		agent.WithTransport(code.hub.NewTransport()),
	)

	// Set up provider
	code.provider = provider.New(
		provider.WithName("openai"),
	)

	messages := []prvdr.Message{
		{
			Role:    "system",
			Content: tweaker.GetSystemPrompt("planner"),
		},
		{
			Role:    "user",
			Content: "Please plan the implementation of a basic HTTP server",
		},
	}

	buf, err := json.Marshal(messages)

	if err != nil {
		fmt.Printf("Error marshalling messages: %v\n", err)
		return
	}

	// Create and send initial task
	task := datura.New(
		datura.WithRole(datura.ArtifactRoleSystem),
		datura.WithScope(datura.ArtifactScopeGeneration),
		datura.WithPayload(buf),
		datura.WithMeta("model", "gpt-4o-mini"),
		datura.WithMeta("temperature", 0.5),
		datura.WithMeta("top_p", 1.0),
		datura.WithMeta("frequency_penalty", 0.0),
		datura.WithMeta("presence_penalty", 0.0),
		datura.WithMeta("stream", false),
	)

	future, release := code.provider.Client().Generate(code.ctx, func(p provider.RPC_generate_Params) error {
		return p.SetContext(*task.Artifact)
	})

	defer release()

	if _, err := future.Struct(); err != nil {
		fmt.Printf("Error generating task: %v\n", err)
		return
	}

	// Start agents
	go code.planner.Run(code.ctx, code.dev.Transport)
	go code.dev.Run(code.ctx, code.planner.Transport)

	// Handle shutdown
	code.shutdown()
}

func (code *Code) shutdown() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for signal or timeout
	select {
	case sig := <-sigChan:
		fmt.Printf("\nReceived signal %v, shutting down...\n", sig)
		code.cancel()
	case <-time.After(30 * time.Second):
		fmt.Println("Timeout reached, shutting down...")
		code.cancel()
	}

	fmt.Println("Shutdown complete")
}
